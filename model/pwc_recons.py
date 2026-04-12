import torch
import torch.nn as nn
# from model import flow_pwc
import model.blocks as blocks
from model.kernel import KernelNet
import torch.nn.functional as F
import math
import model.deconv_fft as deconv_fft
from model.submodules import MSD
from model.submodules import HAB


def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    flow_pretrain_fn = args.pretrain_models_dir + 'network-default.pytorch'
    kernel_pretrain_fn = args.pretrain_models_dir + 'DBVSR_kernel23_30_SFT.pt'
    return PWC_Recons(n_colors=args.n_colors, n_sequence=args.n_sequence, extra_RBS=args.extra_RBS,
                      recons_RBS=args.recons_RBS, n_feat=args.n_feat, n_cond=args.n_cond, est_ksize=args.est_ksize,
                      scale=args.scale, flow_pretrain_fn=flow_pretrain_fn, kernel_pretrain_fn=kernel_pretrain_fn,
                      device=device)

class PWC_Recons(nn.Module):

    def __init__(self, n_colors=3, n_sequence=5, extra_RBS=1, recons_RBS=3, n_feat=128, n_cond=128, est_ksize=13,
                 kernel_size=3, scale=4, flow_pretrain_fn='.', kernel_pretrain_fn='.', device='cuda'):
        super(PWC_Recons, self).__init__()
        print("Creating PWC-Recons Net")

        self.n_sequence = n_sequence
        self.scale = scale

        In_conv = [nn.Conv2d(n_colors, n_feat, kernel_size=3, stride=1, padding=1)]

        Extra_feat = []
        Extra_feat.extend([blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1)
                           for _ in range(extra_RBS)])

        my_Fusion_conv1 = [nn.Conv2d(n_feat * 3, n_feat, kernel_size=3, stride=1, padding=1)]
        my_Fusion_conv4 = [nn.Conv2d(n_feat * 3, n_feat, kernel_size=3, stride=1, padding=1)]
        my_Fusion_conv5 = [nn.Conv2d(n_feat * 3, n_feat, kernel_size=3, stride=1, padding=1)]

        my_Fusion_conv2 = [nn.Conv2d(n_feat * 3, 180, kernel_size=3, stride=1, padding=1)]

        my_Fusion_conv3 = [nn.Conv2d(180, n_feat, kernel_size=3, stride=1, padding=1)]

        Recons_net = []
        Recons_net.extend([blocks.ResBlock_SFT(n_feat, n_cond) for _ in range(recons_RBS)])
        Recons_net.extend([
            blocks.SFTLayer(),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        ])

        Out_conv = [
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(n_feat, n_colors, kernel_size=3, stride=1, padding=1)
        ]

        CondNet = [  # kernel：b 1 13 13 --> feature: b 64 64 64
            nn.Conv2d(n_colors * scale * scale, n_feat, 5, 1, 4, dilation=2), nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_feat, 3, 1, 2, dilation=2), nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_feat, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_cond, 1)
        ]

        Upsample_layers = []  # Recontribution
        for _ in range(int(math.log2(scale))):
            Upsample_layers.append(nn.Conv2d(n_feat, n_feat * 4, 3, 1, 1, bias=True))
            Upsample_layers.append(nn.PixelShuffle(2))       

        self.in_conv = nn.Sequential(*In_conv)   
        self.extra_feat = nn.Sequential(*Extra_feat)
        self.fusion_conv1 = nn.Sequential(*my_Fusion_conv1)  
        self.fusion_conv4 = nn.Sequential(*my_Fusion_conv4)  # channel: 384 --> 128
        self.fusion_conv5 = nn.Sequential(*my_Fusion_conv5)  # channel: 384 --> 128
        # self.fusion_conv = nn.Sequential(*Fusion_conv)
        self.recons_net = nn.Sequential(*Recons_net)
        self.out_conv = nn.Sequential(*Out_conv)
        self.upsample_layers = nn.Sequential(*Upsample_layers)
        # self.flow_net = flow_pwc.Flow_PWC(pretrain_fn=flow_pretrain_fn, device=device)
        self.kernel_net = KernelNet(ksize=est_ksize)
        self.cond_net = nn.Sequential(*CondNet)
        self.msd_align = MSD(nf=128, groups=8, dilation=1)
        self.msdeformable_fusion = HAB()

        self.fusion_conv2 = nn.Sequential(*my_Fusion_conv2)  # channel: 384 --> 180
        self.fusion_conv3 = nn.Sequential(*my_Fusion_conv3)  # channel: 180 --> 128


        if kernel_pretrain_fn != '.':
            self.kernel_net.load_state_dict(torch.load(kernel_pretrain_fn))
            print('Loading KernelNet pretrain model from {}'.format(kernel_pretrain_fn))

        my_Fusion_conv8 = [nn.Conv2d(3, n_feat, kernel_size=3, stride=1, padding=1)]
        my_Fusion_conv9 = [nn.Conv2d(n_feat, 3, kernel_size=3, stride=1, padding=1)]
        self.fusion_conv8 = nn.Sequential(*my_Fusion_conv8)  # channel: 384 --> 180
        self.fusion_conv9 = nn.Sequential(*my_Fusion_conv9)  # channel: 180 --> 128
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.upsample_conv = nn.Conv2d(
            n_feat,
            n_feat * (scale ** 2), 
            kernel_size=3,
            padding=1
        )
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, input_dict):
        x = input_dict['x']
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
        frame_feat_list = [self.extra_feat(self.in_conv(frame)) for frame in frame_list]  

        kernel_list = [self.kernel_net(f) for f in frame_list]  #  [8 1 13 13]  *  5
        deconv = deconv_fft.deconv_batch(   # [8 3 256 256]
            frame_list[self.n_sequence // 2],
            kernel_list[self.n_sequence // 2],
            self.scale
        )
        # print(666, deconv.size())
        deconv_S2D = self.spatial2depth(deconv, self.scale)   
        cond = self.cond_net(deconv_S2D)   # [8 48 64 64]  -->  [8 128 64 64]

        base = frame_list[self.n_sequence // 2]  
        # base = F.interpolate(base, scale_factor=self.scale, mode='bilinear', align_corners=False)  # [8 3 64 64]  -->  [8 3 256 256]
        base1 = self.leaky_relu(self.fusion_conv8(base))
        base2 = self.upsample_conv(base1)
        base3 = self.fusion_conv9(self.pixel_shuffle(base2))




        warped_feat_list = []

        # slide window 1
        warped_feat_01 = self.msd_align(frame_feat_list[0], frame_feat_list[1])   # [8 128 64 64]
        warped_feat_21 = self.msd_align(frame_feat_list[2], frame_feat_list[1])   # [8 128 64 64]
        warped_feat_1 = self.fusion_conv1(torch.cat([warped_feat_01, frame_feat_list[1], warped_feat_21],dim=1))  # [8 128 64 64]
        warped_feat_list.append(warped_feat_1)

        # slide window 2
        warped_feat_12 = self.msd_align(frame_feat_list[1], frame_feat_list[2])
        warped_feat_32 = self.msd_align(frame_feat_list[3], frame_feat_list[2])
        warped_feat_2 = self.fusion_conv4(torch.cat([warped_feat_12, frame_feat_list[2], warped_feat_32],dim=1))
        warped_feat_list.append(warped_feat_2)

        # slide window 3
        warped_feat_23 = self.msd_align(frame_feat_list[2], frame_feat_list[3])
        warped_feat_43 = self.msd_align(frame_feat_list[4], frame_feat_list[3])
        warped_feat_3 = self.fusion_conv5(torch.cat([warped_feat_23, frame_feat_list[3], warped_feat_43],dim=1))
        warped_feat_list.append(warped_feat_3)



        feat_to_msd_fusion = self.fusion_conv2(torch.cat(warped_feat_list, dim=1))


        fusion_feat1 = self.msdeformable_fusion(feat_to_msd_fusion)  # [8 128 64 64]
        fusion_feat = self.fusion_conv3(fusion_feat1)

        recons_feat = self.recons_net((fusion_feat, cond))  # [8 128 64 64]
        recons = self.out_conv(self.upsample_layers(recons_feat))  # [8 3 256 256]
        # recons = recons + base  # [8 3 256 256]
        recons = recons + base3  # [8 3 256 256]


        mid_loss = None

        return {
            'recons': recons,
            'kernel_list': kernel_list
               }, mid_loss

    def spatial2depth(self, spatial, scale):  
        depth_list = []
        for i in range(scale):
            for j in range(scale):
                depth_list.append(spatial[:, :, i::scale, j::scale])
        depth = torch.cat(depth_list, dim=1)
        return depth


