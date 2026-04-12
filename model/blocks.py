import torch
import torch.nn as nn
import torch.nn.functional as F


###############################
# common
###############################

def get_act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def get_norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batchnorm':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instancenorm':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def get_same_padding(kernel_size, dilation):   
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def get_sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
        else:
            raise Exception("Unsupport module type [{:s}]".format(type(module)))
    return nn.Sequential(*modules)


class CommonConv(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
                 pad_type='same', norm_type=None, act_type=None, mode='CNA'):
        super(CommonConv, self).__init__()

        mode = mode.upper()
        pad_type = pad_type.lower()
        norm_type = norm_type.lower()
        act_type = act_type.lower()

        if pad_type == 'zero':
            padding = 0
        elif pad_type == 'same':
            padding = get_same_padding(kernel_size, dilation)
        else:
            raise NotImplementedError('padding type [{:s}] is not found'.format(pad_type))
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias, groups=groups)

        self.act = get_act(act_type=act_type) if act_type else None

        if mode == "CNA":
            self.norm = get_norm(norm_type=norm_type, nc=out_nc) if norm_type else None
        elif mode == "NAC":
            self.norm = get_norm(norm_type=norm_type, nc=in_nc) if norm_type else None
        else:
            raise NotImplementedError('convolution mode [{:s}] is not found'.format(mode))

        self.mode = mode
        self.pad_type = pad_type
        self.norm_type = norm_type
        self.act_type = act_type

    def forward(self, x):
        if self.mode == "CNA":
            x = self.conv(x)
            x = self.norm(x) if self.norm else x
            x = self.act(x) if self.act else x
        elif self.mode == "NAC":
            x = self.norm(x) if self.norm else x
            x = self.act(x) if self.act else x
            x = self.conv(x)
        else:
            x = x
        return x

class CA_layer(nn.Module):  
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(x[1][:, :, None, None])

        return x[0] * att
###############################
# ResNet 
###############################

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,    
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

        self.res_translate = None    
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out


###############################
# Residual Dense Network 
###############################

class DenseConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, act_type='relu'):
        super(DenseConv, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                              padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.act = get_act(act_type=act_type) if act_type else None

    def forward(self, x):
        output = self.act(self.conv(x))
        return torch.cat((x, output), 1)


class RDB(nn.Module):

    def __init__(self, inplanes, planes, midplanes=None, n_conv=6, kernel_size=3, stride=1, dilation=1):
        super(RDB, self).__init__()

        if not midplanes:
            midplanes = inplanes

        layers = []
        for i in range(n_conv):
            layers.append(DenseConv(inplanes + i * midplanes, midplanes,
                                    kernel_size=kernel_size, stride=1, dilation=dilation))
        layers.append(nn.Conv2d(inplanes + n_conv * midplanes, planes, kernel_size=1, stride=stride))
        self.layers = nn.Sequential(*layers)

        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x

        out = self.layers(x)

        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out


###############################
# U-net 
###############################

class UnetBottleneck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, dilation=1, act_type='relu'):
        super(UnetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.act = get_act(act_type=act_type) if act_type else None

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        return out


class UnetDownBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, dilation=1, act_type='relu'):
        super(UnetDownBlock, self).__init__()
        self.down = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1)
        self.conv = UnetBottleneck(inplanes, planes, kernel_size=kernel_size, dilation=dilation, act_type=act_type)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class UnetUpBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, dilation=1, act_type='relu'):
        super(UnetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(inplanes, inplanes // 2, kernel_size=4, stride=2, padding=1)
        self.conv = UnetBottleneck(inplanes, planes, kernel_size=kernel_size, dilation=dilation, act_type=act_type)

    def forward(self, x1, x2):
        x1_up = self.up(x1)
        out = self.conv(torch.cat([x2, x1_up], dim=1))
        return out


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=128):
        super(TSAFusion, self).__init__()
        # self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        # self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn6 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.spatial_attn7 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn8 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn9 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        # self.swin1 = SwinPCCA(c1=256, c2=128, e=1)
        self.swin2 = SwinPCCA(c1=128, c2=128, e=1)
        # self.conv1 = nn.Conv2d(256, 128, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, fused, cond):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        # fusion
        b, c, h, w = fused.size()
        feat = fused
        aligned_feat = torch.stack([fused, cond], dim=1)   # [8 2 128 64 64]
        aligned_feat = aligned_feat.view(b, -1, h, w)    # [8 256 64 64]

        # print(aligned_feat.size())

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))   # [8 128 64 64]
        attn_max = self.max_pool(attn)   # [8 128 32 32]
        attn_avg = self.avg_pool(attn)   # [8 128 32 32]
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))   # [8 128 32 32]
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))   # [8 128 32 32]
        attn_max = self.max_pool(attn_level)   # [8 128 16 16]
        attn_avg = self.avg_pool(attn_level)   # [8 128 16 16]
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))   # [8 128 16 16]
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))   # [8 128 16 16]
        attn_level = self.upsample(attn_level)    # [8 128 32 32]

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level   # [8 128 32 32]
        attn = self.lrelu(self.spatial_attn4(attn))    # [8 128 32 32]
        attn = self.upsample(attn)   # [8 128 64 64]
        attn = self.spatial_attn5(attn)    # [8 128 64 64]

        attn_add = self.spatial_attn_add2(      # [8 128 64 64]
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)   # [8 128 64 64]
        # attn3 = self.swin2(self.lrelu(self.spatial_attn6(feat)))
        # attn3 = torch.cat((attn3, feat), dim=1)
        # attn3 = self.lrelu(self.spatial_attn7(attn3))

        # attn3 = self.lrelu(self.spatial_attn8(self.swin2(self.lrelu(self.spatial_attn6(feat)))))

        attn3 = self.lrelu(self.spatial_attn8(self.swin2(self.lrelu(self.spatial_attn6(feat)))))
        attn3 = feat + attn3
        attn3 = self.lrelu(self.spatial_attn9(attn3))

        feat = feat * attn * 2 + attn_add + attn3  # [8 128 64 64]
        return feat

class SFTLayer(nn.Module):
    def __init__(self):  # 128, 64
        super(SFTLayer, self).__init__()
        self.sa = TSAFusion()
    def forward(self, x):
        out = self.sa(x[0], x[1])   # [8 128 64 64]
        return out



class ResBlock_SFT(nn.Module):              
    def __init__(self, n_feat, n_cond):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions


######################

class SwinPCCA(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, e):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(SwinPCCA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.eca = ECA(c_)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.cv1(x)
        x = self.eca(x)
        out = self.sa(x) * x
        return out

# ECA attention
class ECA(nn.Module):

    def __init__(self, c1, k_size=3):          
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):                 
        
        super(SpatialAttention, self).__init__()

        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):                               
        avg_out = torch.mean(x, dim=1, keepdim=True)    
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        x = torch.cat([avg_out, max_out], dim=1)        
        x = self.conv1(x)                               
        return self.sigmoid(x)

class Conv(nn.Module):
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, padding=0, dilation=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False, dilation=1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x): 
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x): 
        return self.act(self.conv(x))

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

###############################
# GCA
###############################
class GCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # K = d*(k_size-1)+1
        # (H - k_size + 2padding)/stride + 1
        # (5,1)-->(7,3)
        # self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # K=5, 64-5+4+1=64
        # self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        # (3,1)-->(5,2)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  #
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2) # K=9, 64-9+8 + 1

        # (5,1)-->(7,4)
        # self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  #
        # self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=12, groups=dim, dilation=4) # K=25, 64-25+2*12 + 1

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)              # Spatical Attention
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn
