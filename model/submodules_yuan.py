#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import math
import torch.utils.checkpoint as checkpoint
import numbers
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_

from einops import rearrange


from os import terminal_size
# from time import thread_time_ns               ###

# from utils import log
from warnings import simplefilter
import torch.nn as nn
import torch
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import numpy as np
# from config import cfg


# from einops import rearrange, repeat              ###
# from einops.layers.torch import Rearrange         ###
from torch import einsum, rand
from model.ops.modules.ms_deform_attn import MSDeformAttn, MSDeformAttn_Fusion
# from models.position_encoding import PositionEmbeddingSine
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
# from utils.network_utils import warp
from torch.autograd import Variable
from model.DCNv2.dcn_v2 import DCN_sep

from model import flow_pwc
from model.blocks import SwinPCCA

# GCD 改进应用(全局上下文注意力模块)
class MSD_11(nn.Module):
    def __init__(self, input=128, dim=180):
        super(MSD, self).__init__()
        # K = d*(k_size-1)+1
        # (H - k_size + 2padding)/stride + 1
        # (5,1)-->(7,3)
        # self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # K=5, 64-5+4+1=64
        # self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.input = nn.Conv2d(input * 2, dim, 3, 1, 1)
        # (3,1)-->(5,2)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  #
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2) # K=9, 64-9+8 + 1
        self.conv_spatial2 = nn.Conv2d(dim, dim, 7, stride=1, padding=6, groups=dim, dilation=2)


        # (5,1)-->(7,4)
        # self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  #
        # self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=12, groups=dim, dilation=4) # K=25, 64-25+2*12 + 1

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv3 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(dim * 3 // 2, input, 1, 1)
        # self.conv = nn.Conv2d(dim // 2, dim, 1)

        self.dcnpack = DCN_sep(input, input, 3, stride=1, padding=1, dilation=1, deformable_groups=8)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        x = self.input(offset)

        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn3 = self.conv_spatial2(attn2)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn3 = self.conv3(attn3)

        attn = torch.cat([attn1, attn2, attn3], dim=1)
        offset2 = self.lrelu(self.conv_squeeze(attn))
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset2))
        # avg_attn = torch.mean(attn, dim=1, keepdim=True)              # Spatical Attention
        # max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        # agg = torch.cat([avg_attn, max_attn], dim=1)
        # sig = self.conv_squeeze(agg).sigmoid()
        # attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        # attn = self.conv(attn)
        return aligned_fea

# 参考TTST深度卷积Chunk
class MSD_10(nn.Module):
    def __init__(self, input=128, nf=180, groups=180, bias=False):
        super(MSD, self).__init__()
        self.input = nn.Conv2d(input * 2, nf, 3, 1, 1, bias=bias)
        # self.input0 = nn.Conv2d(nf, input, 3, 1, 1, bias=bias)

        self.mul_scale1 = nn.Conv2d(nf, nf, 3, 1, 1, groups=groups, bias=bias)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf, nf, 5, 1, 2, groups=groups, bias=bias)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf, 7, 1, 3, groups=groups, bias=bias)   # (64-7+2*3)/1+1=64

        self.mul_scale4 = nn.Conv2d(nf, nf, 3, 1, 1, groups=groups, bias=bias)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale5 = nn.Conv2d(nf, nf, 5, 1, 2, groups=groups, bias=bias)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale6 = nn.Conv2d(nf, nf, 7, 1, 3, groups=groups, bias=bias)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        # self.offset_conv2 = nn.Conv2d(3 * nf, nf, 1, 1, bias=bias)
        self.offset_conv3 = nn.Conv2d(3 * nf, input, 1, 1, bias=bias)

        self.dcnpack = DCN_sep(input, input, 3, stride=1, padding=1, dilation=1, deformable_groups=8)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.ReLU()

    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        x = self.relu(self.input(offset))
        # x0 = self.relu(self.input0(x))

        fea11, fea12, fea13 = self.relu(self.mul_scale1(x)).split(60, dim=1)
        fea21, fea22, fea23 = self.relu(self.mul_scale2(x)).split(60, dim=1)
        fea31, fea32, fea33 = self.relu(self.mul_scale3(x)).split(60, dim=1)

        fea1 = torch.cat([fea11, fea21, fea31], dim=1)
        fea2 = torch.cat([fea12, fea22, fea32], dim=1)
        fea3 = torch.cat([fea13, fea23, fea33], dim=1)

        fea4 = self.relu(self.mul_scale4(fea1))
        fea5 = self.relu(self.mul_scale5(fea2))
        fea6 = self.relu(self.mul_scale6(fea3))

        fea = torch.cat([fea4, fea5, fea6], dim=1)
        fea = self.relu(self.offset_conv3(fea))

        # offset_final = x0 + fea

        aligned_fea = self.relu(self.dcnpack(neibor_fea, fea))
        return aligned_fea

# 密集残差RDB应用(最后一层有无Relu，Relu Or lRelu)   --->  1 * 1 Conv  +   offset6(relu)
class MSD_9(nn.Module):


    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)   # 用来卷积concat后的特征
        self.mul_scale1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64

        self.mul_scale4 = nn.Conv2d(nf * 4, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale5 = nn.Conv2d(nf * 4, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale6 = nn.Conv2d(nf * 4, nf, 7, 1, 3, bias=True)

        # self.mul_scale7 = nn.Conv2d(nf * 8, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        # self.mul_scale8 = nn.Conv2d(nf * 8, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        # self.mul_scale9 = nn.Conv2d(nf * 8, nf, 7, 1, 3, bias=True)
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(13 * nf, nf, 1, 1)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作

        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.Relu = nn.ReLU(inplace=True)


    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        # offset1 = self.lrelu(self.offset_conv1(offset))
        offset1 = self.Relu(self.offset_conv1(offset))
        mul_scale_fea1 = self.Relu(self.mul_scale1(offset1))
        mul_scale_fea2 = self.Relu(self.mul_scale2(offset1))
        mul_scale_fea3 = self.Relu(self.mul_scale3(offset1))
        offset2 = torch.cat([offset1, mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)

        mul_scale_fea4 = self.Relu(self.mul_scale4(offset2))
        mul_scale_fea5 = self.Relu(self.mul_scale5(offset2))
        mul_scale_fea6 = self.Relu(self.mul_scale6(offset2))
        offset3 = torch.cat([offset1, offset2, mul_scale_fea4, mul_scale_fea5, mul_scale_fea6], dim=1)

        # mul_scale_fea7 = self.Relu(self.mul_scale7(offset3))
        # mul_scale_fea8 = self.Relu(self.mul_scale8(offset3))
        # mul_scale_fea9 = self.Relu(self.mul_scale9(offset3))
        # offset4 = torch.cat([offset1, offset2, offset3, mul_scale_fea7, mul_scale_fea8, mul_scale_fea9], dim=1)

        # offset5 = torch.cat([offset1, offset2, offset3, offset4], dim=1)
        # offset6 = self.offset_conv2(offset5)

        offset4 = torch.cat([offset1, offset2, offset3], dim=1)
        offset6 = self.Relu(self.offset_conv2(offset4))

        offset_final = offset1 + offset6
        # aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset_final))
        aligned_fea = self.Relu(self.dcnpack(neibor_fea, offset_final))

        return aligned_fea

# 改进TDAN_yuan(conv重复)
class MSD_TDAN_multilayer(nn.Module):
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)   # 用来卷积concat后的特征
        mul_scale1 = [
            nn.Conv2d(nf, nf, 1, 1, bias=True),   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),  # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
            nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # (64-7+2*3)/1+1=64
        ]
        self.mul_scale1 = nn.Sequential(*mul_scale1)
        mul_scale2 = [
            nn.Conv2d(64, 64, 1, 1, bias=True),   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
            nn.Conv2d(64, 64, 5, 1, 2, bias=True)   # (64-7+2*3)/1+1=64
        ]
        self.mul_scale2 = nn.Sequential(*mul_scale2)
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        # self.offset_conv2 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作
        self.dcnpack2 = DCN_sep(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.cr = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        # self.off2d_1 = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        # self.dconv_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        # self.off2d_2 = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        # self.deconv_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        # self.off2d_3 = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        # self.deconv_3 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        # self.off2d = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        # self.dconv = DCN_sep(nf, nf, (3, 3), stride=1, padding=1, dilation=1, deformable_groups=groups)
        # self.recon_lr = nn.Conv2d(64, 128, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer = self.make_layer(Res_Block, 5)
        self.recon_lr = nn.Conv2d(64, 3, 3, padding=1, bias=True)

        self.cr = nn.Conv2d(128, 128, 3, padding=1, bias=True)
        self.off2d_1 = nn.Conv2d(128, 128, 3, padding=1, bias=True)
        # self.dconv_1 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_2 = nn.Conv2d(128, 128, 3, padding=1, bias=True)
        # self.deconv_2 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_3 = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        # self.deconv_3 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        # self.dconv = ConvOffset2d(64, 64, (3, 3), padding=(1, 1), num_deformable_groups=8)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def align(self, x, x_center):
        y = []
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        ref = x[:, center, :, :, :].clone()
        for i in range(num):
            if i == center:
                y.append(x_center.unsqueeze(1))
                continue
            supp = x[:, i, :, :, :]
            fea = torch.cat([ref, supp], dim=1)
            # fea = self.cr(fea)
            # feature trans
            offset1 = self.mul_scale1(fea)
            fea = (self.dcnpack(fea, offset1))
            offset2 = self.mul_scale1(fea)
            fea = (self.dcnpack(fea, offset2))
            offset3 = self.mul_scale1(fea)
            fea = (self.dcnpack(fea, offset3))
            offset = self.off2d_3(fea)
            fea = (self.dcnpack2(supp.contiguous(), offset))
            offset4 = self.mul_scale2(fea)
            fea = self.dcnpack2(fea, offset4)
            offset5 = self.mul_scale2(fea)
            fea = self.dcnpack2(fea, offset5)
            offset6 = self.mul_scale2(fea)
            aligned_fea = self.dcnpack2(fea, offset6)
            im = self.recon_lr(aligned_fea).unsqueeze(1)
            y.append(im)
        y = torch.cat(y, dim=1)
        return y

    def forward(self, x):
        batch_size, num, ch, w, h = x.size()  # 5 video frames
        # center frame interpolation
        center = num // 2
        # extract features
        y = x.view(-1, ch, w, h)
        # y = y.unsqueeze(1)
        out = self.relu(self.conv_first(y))
        x_center = x[:, center, :, :, :]
        out = self.residual_layer(out)
        out = out.view(batch_size, num, -1, w, h)
        # align supporting frames
        lrs = self.align(out, x_center) # motion alignments
        y = lrs.view(batch_size, -1, w, h)
        return y

# 恢复TDAN(MSD_yuan)
class MSD_TDAN(nn.Module):
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.dcnpack1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作
        self.dcnpack2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作
        self.dcnpack3 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作

        self.dcnpack4 = DCN_sep(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作
        self.dcnpack5 = DCN_sep(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作
        self.dcnpack6 = DCN_sep(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作
        self.dcnpack7 = DCN_sep(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作


        self.relu = nn.ReLU(inplace=True)
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer = self.make_layer(Res_Block, 5)
        self.recon_lr = nn.Conv2d(64, 3, 3, padding=1, bias=True)

        self.cr = nn.Conv2d(128, 128, 3, padding=1, bias=True)
        self.off2d_1 = nn.Conv2d(128, 128, 3, padding=1, bias=True)
        # self.dconv_1 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_2 = nn.Conv2d(128, 128, 3, padding=1, bias=True)
        self.off2d_3 = nn.Conv2d(128, 128, 3, padding=1, bias=True)
        # self.deconv_2 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_4 = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        # self.deconv_3 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        self.off2d_5 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.off2d_6 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.off2d_7 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        # self.dconv = ConvOffset2d(64, 64, (3, 3), padding=(1, 1), num_deformable_groups=8)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def align(self, x, x_center):
        y = []
        batch_size, num, ch, w, h = x.size()
        center = num // 2
        ref = x[:, center, :, :, :].clone()
        for i in range(num):
            if i == center:
                y.append(x_center.unsqueeze(1))
                continue
            supp = x[:, i, :, :, :]
            fea = torch.cat([ref, supp], dim=1)
            fea = self.cr(fea)
            # feature trans
            offset1 = self.off2d_1(fea)
            fea = (self.dcnpack1(fea, offset1))
            offset2 = self.off2d_2(fea)
            fea = (self.dcnpack2(fea, offset2))
            offset3 = self.off2d_3(fea)
            fea = (self.dcnpack3(fea, offset3))
            offset = self.off2d_4(fea)
            fea = (self.dcnpack4(supp.contiguous(), offset))
            offset4 = self.off2d_5(fea)
            fea = self.dcnpack5(fea, offset4)
            offset5 = self.off2d_6(fea)
            fea = self.dcnpack6(fea, offset5)
            offset6 = self.off2d_7(fea)
            aligned_fea = self.dcnpack7(fea, offset6)
            im = self.recon_lr(aligned_fea).unsqueeze(1)
            y.append(im)
        y = torch.cat(y, dim=1)
        return y

    def forward(self, x):
        batch_size, num, ch, w, h = x.size()  # 5 video frames
        # center frame interpolation
        center = num // 2
        # extract features
        y = x.view(-1, ch, w, h)
        # y = y.unsqueeze(1)
        out = self.relu(self.conv_first(y))
        x_center = x[:, center, :, :, :]
        out = self.residual_layer(out)
        out = out.view(batch_size, num, -1, w, h)
        # align supporting frames
        lrs = self.align(out, x_center) # motion alignments
        y = lrs.view(batch_size, -1, w, h)
        return y

# 最后加入注意力机制加权（待测）
class MSD_8(nn.Module):
    '''
    Alignment with multi-scale deformable conv and attention mechanism
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)
        self.mul_scale1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.mul_scale2 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)
        self.mul_scale3 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)
        self.offset_conv2 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)
        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.attention = nn.Sequential(
            nn.Conv2d(nf, nf//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf//8, 1, 1),
            nn.Sigmoid()
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.offset_conv1(offset))
        mul_scale_fea1 = self.lrelu(self.mul_scale1(offset1))
        mul_scale_fea2 = self.lrelu(self.mul_scale2(offset1))
        mul_scale_fea3 = self.lrelu(self.mul_scale3(offset1))
        offset2 = torch.cat([mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)
        offset2 = self.lrelu(self.offset_conv2(offset2))
        offset_final = offset1 + offset2
        attention_weight = self.attention(offset_final)  # 使用注意力机制加权
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset_final) * attention_weight)  # 利用注意力权重调整特征图
        return aligned_fea

# 加入归一化层和两个DconV（待测，目前加入BN层效果不好）
class MSD_7(nn.Module):
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(nf) # 新增批归一化层
        self.mul_scale1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(nf) # 新增批归一化层
        self.mul_scale2 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)
        self.bn3 = nn.BatchNorm2d(nf) # 新增批归一化层
        self.mul_scale3 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)
        self.bn4 = nn.BatchNorm2d(nf) # 新增批归一化层
        self.offset_conv2 = nn.Conv2d(4 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation) # 将3个多尺度特征和之前的offset聚合起来，新增了一个多尺度特征提取层
        self.bn5 = nn.BatchNorm2d(nf) # 新增批归一化层
        self.dcnpack1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.dcnpack2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups) # 新增了一个可变形卷积层
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.bn1(self.offset_conv1(offset))) # 添加批归一化层
        mul_scale_fea1 = self.lrelu(self.bn2(self.mul_scale1(offset1)))
        mul_scale_fea2 = self.lrelu(self.bn3(self.mul_scale2(offset1)))
        mul_scale_fea3 = self.lrelu(self.bn4(self.mul_scale3(offset1)))
        offset2 = torch.cat([mul_scale_fea1, mul_scale_fea2, mul_scale_fea3, offset1], dim=1) # 将之前的offset也加入到聚合过程中
        offset2 = self.lrelu(self.bn5(self.offset_conv2(offset2)))
        offset_final = offset1 + offset2
        aligned_fea1 = self.lrelu(self.dcnpack1(neibor_fea, offset_final)) # 使用第一个可变形卷积层
        aligned_fea2 = self.lrelu(self.dcnpack2(aligned_fea1, offset_final)) # 使用第二个可变形卷积层
        return aligned_fea2

# 原+4层卷积+并联2层,比MSD2多了一个残差（效果变好）
class MSD_6(nn.Module):
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)   # 用来卷积concat后的特征
        self.mul_scale0 = nn.Conv2d(nf, nf, 1, 1, bias=True)  # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64
        self.mul_scale4 = nn.Conv2d(nf, nf, 1, 1, bias=True)  # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale6 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale7 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(4 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.offset_conv3 = nn.Conv2d(4 * nf, nf, 1, 1, bias=True, dilation=dilation)

        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.offset_conv1(offset))
        mul_scale_fea0 = self.lrelu(self.mul_scale0(offset1))
        mul_scale_fea1 = self.lrelu(self.mul_scale1(offset1))
        mul_scale_fea2 = self.lrelu(self.mul_scale2(offset1))
        mul_scale_fea3 = self.lrelu(self.mul_scale3(offset1))
        offset2 = torch.cat([mul_scale_fea0, mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)
        offset2 = self.lrelu(self.offset_conv2(offset2))
        # offset_final = offset1 + offset2

        mul_scale_fea4 = self.lrelu(self.mul_scale4(offset2))
        mul_scale_fea5 = self.lrelu(self.mul_scale5(offset2))
        mul_scale_fea6 = self.lrelu(self.mul_scale6(offset2))
        mul_scale_fea7 = self.lrelu(self.mul_scale7(offset2))
        offset3 = torch.cat([mul_scale_fea4, mul_scale_fea5, mul_scale_fea6, mul_scale_fea7], dim=1)
        offset4 = self.lrelu(self.offset_conv3(offset3))

        offset_final = offset1 + offset4 + offset2     # 比MSD2多了一个残差
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset_final))

        return aligned_fea

# 光流法 指导 DconV(待测DCNv3)
class MSD_5(nn.Module):
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        # self.dcnpack = DCNv3(nf, kernel_size=1, stride=1, padding=1, dilation=1, group=8)   # 用得到的offset指导可变形操作
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, neibor_fea, target_fea):
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, target_fea))
        return aligned_fea

# 加入最大池化，类似于金字塔（效果不好）
class MSD_4(nn.Module):
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)   # 用来卷积concat后的特征
        self.mul_scale1 = nn.Conv2d(nf, nf//2, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf//2, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf * 2, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # 不同尺度的卷积层
        self.conv4_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv4_3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4_5 = nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2)

    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.offset_conv1(offset))
        mul_scale_fea1 = self.mul_scale1(offset1)
        pool1_output = nn.MaxPool2d(kernel_size=2, stride=2)(mul_scale_fea1)

        mul_scale_fea2 = self.mul_scale2(pool1_output)
        pool2_output = nn.MaxPool2d(kernel_size=2, stride=2)(mul_scale_fea2)

        mul_scale_fea3 = self.mul_scale3(pool2_output)
        pool3_output = nn.MaxPool2d(kernel_size=2, stride=2)(mul_scale_fea3)

        upsample1_output = F.interpolate(pool3_output, scale_factor=8, mode='bilinear', align_corners=False)

        # 不同尺度的卷积层
        conv4_1_output = self.conv4_1(upsample1_output)
        conv4_3_output = self.conv4_3(upsample1_output)
        conv4_5_output = self.conv4_5(upsample1_output)

        offset2 = torch.cat([conv4_1_output, conv4_3_output, conv4_5_output], dim=1)
        offset2 = self.lrelu(self.offset_conv2(offset2))
        # offset_final = offset1 + offset2
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset2))

        return aligned_fea

# 添加两个额外的尺度变换操作（待测，感觉一般）
class MSD_3(nn.Module):
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)   # 用来卷积concat后的特征
        self.mul_scale1 = nn.Conv2d(nf, nf//2, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf//2, nf//4, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf//4, nf//8, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(112, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.offset_conv1(offset))
        mul_scale_fea1 = self.mul_scale1(offset1)
        pool1_output = nn.MaxPool2d(kernel_size=2)(mul_scale_fea1)

        mul_scale_fea2 = self.mul_scale2(pool1_output)
        pool2_output = nn.MaxPool2d(kernel_size=2)(mul_scale_fea2)

        mul_scale_fea3 = self.mul_scale3(pool2_output)

        # 添加额外的尺度变换操作
        upsample1_output = nn.Upsample(scale_factor=2, mode='bilinear')(mul_scale_fea2)
        upsample2_output = nn.Upsample(scale_factor=4, mode='bilinear')(mul_scale_fea3)

        offset2 = torch.cat([mul_scale_fea1, upsample1_output, upsample2_output], dim=1)
        offset2 = self.lrelu(self.offset_conv2(offset2))
        # offset_final = offset1 + offset2
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset2))

        return aligned_fea

# 比MSD2多了BN层,未引入残差（效果下降）
class MSD_2_1(nn.Module):
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)
        self.mul_scale0 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.mul_scale1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.mul_scale2 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)
        self.mul_scale3 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)

        self.offset_conv2 = nn.Conv2d(4 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)
        self.mul_scale4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.mul_scale5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.mul_scale6 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)
        self.mul_scale7 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)

        self.offset_conv3 = nn.Conv2d(4 * nf, nf, 1, 1, bias=True, dilation=dilation)

        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.bn = nn.BatchNorm2d(nf)

    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.offset_conv1(offset))
        mul_scale_fea0 = self.lrelu(self.mul_scale0(offset1))
        mul_scale_fea1 = self.lrelu(self.mul_scale1(offset1))
        mul_scale_fea2 = self.lrelu(self.mul_scale2(offset1))
        mul_scale_fea3 = self.lrelu(self.mul_scale3(offset1))
        offset2 = torch.cat([mul_scale_fea0, mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)
        offset2 = self.lrelu(self.offset_conv2(offset2))

        mul_scale_fea4 = self.lrelu(self.mul_scale4(offset2))
        mul_scale_fea5 = self.lrelu(self.mul_scale5(offset2))
        mul_scale_fea6 = self.lrelu(self.mul_scale6(offset2))
        mul_scale_fea7 = self.lrelu(self.mul_scale7(offset2))
        offset3 = torch.cat([mul_scale_fea4, mul_scale_fea5, mul_scale_fea6, mul_scale_fea7], dim=1)
        offset4 = self.lrelu(self.offset_conv3(offset3))

        aligned_fea = self.dcnpack(neibor_fea, offset4)
        aligned_fea = self.bn(aligned_fea)
        aligned_fea = self.lrelu(aligned_fea)

        return aligned_fea

# 原+4层卷积+并联2层(未引入残差)(测试DCNv3)
class MSD_2(nn.Module):
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)   # 用来卷积concat后的特征
        self.mul_scale0 = nn.Conv2d(nf, nf, 1, 1, bias=True)  # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64

        self.mul_scale4 = nn.Conv2d(nf, nf, 1, 1, bias=True)  # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale6 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale7 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(4 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.offset_conv3 = nn.Conv2d(4 * nf, nf, 1, 1, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset

        # self.dcnpack = DCNv3(nf, kernel_size=3, stride=1, pad=1, dilation=1, group=8)
        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.offset_conv1(offset))
        mul_scale_fea0 = self.lrelu(self.mul_scale0(offset1))
        mul_scale_fea1 = self.lrelu(self.mul_scale1(offset1))
        mul_scale_fea2 = self.lrelu(self.mul_scale2(offset1))
        mul_scale_fea3 = self.lrelu(self.mul_scale3(offset1))
        offset2 = torch.cat([mul_scale_fea0, mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)
        offset2 = self.lrelu(self.offset_conv2(offset2))

        mul_scale_fea4 = self.lrelu(self.mul_scale4(offset2))
        mul_scale_fea5 = self.lrelu(self.mul_scale5(offset2))
        mul_scale_fea6 = self.lrelu(self.mul_scale6(offset2))
        mul_scale_fea7 = self.lrelu(self.mul_scale7(offset2))
        offset3 = torch.cat([mul_scale_fea4, mul_scale_fea5, mul_scale_fea6, mul_scale_fea7], dim=1)
        offset4 = self.lrelu(self.offset_conv3(offset3))

        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset4))

        aligned_fea = torch.cat((neibor_fea, aligned_fea),dim=1)

        # offset4 = offset4.permute(0, 2, 3, 1)
        # aligned_fea = self.lrelu(self.dcnpack(offset4))
        # aligned_fea = aligned_fea.permute(0, 3, 1, 2)
        return aligned_fea

#################### 135 357 579 相并联（ + ChannelAttention)
class MSD_1_2_CA(nn.Module):
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)   # 用来卷积concat后的特征
        self.mul_scale1 = nn.Conv2d(nf, nf, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变

        self.mul_scale4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale5 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale6 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变

        self.mul_scale7 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale8 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64
        self.mul_scale9 = nn.Conv2d(nf, nf, 9, 1, 4, bias=True)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.offset_conv3 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.offset_conv4 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.offset_conv5 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.ca = ChannelAttention(num_feat=nf, squeeze_factor=30)
        # self.output = nn.Conv2d(4 * nf, nf, 3, 1, 1)
        # self.output = nn.Conv2d(4 * nf, nf, 1, 1)
        # self.selfattention = SelfAttention(embed_dim=64 * 64),


    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.offset_conv1(offset))
        mul_scale_fea1 = self.lrelu(self.mul_scale1(offset1))
        mul_scale_fea2 = self.lrelu(self.mul_scale2(offset1))
        mul_scale_fea3 = self.lrelu(self.mul_scale3(offset1))
        offset2 = torch.cat([mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)
        offset2 = self.ca(self.offset_conv2(offset2))

        mul_scale_fea4 = self.lrelu(self.mul_scale4(offset1))
        mul_scale_fea5 = self.lrelu(self.mul_scale5(offset1))
        mul_scale_fea6 = self.lrelu(self.mul_scale6(offset1))
        offset3 = torch.cat([mul_scale_fea4, mul_scale_fea5, mul_scale_fea6], dim=1)
        offset3 = self.ca(self.offset_conv3(offset3))

        mul_scale_fea7 = self.lrelu(self.mul_scale7(offset1))
        mul_scale_fea8 = self.lrelu(self.mul_scale8(offset1))
        mul_scale_fea9 = self.lrelu(self.mul_scale9(offset1))
        offset4 = torch.cat([mul_scale_fea7, mul_scale_fea8, mul_scale_fea9], dim=1)
        offset4 = self.ca(self.offset_conv4(offset4))
        offset_final = self.lrelu(self.offset_conv5(torch.cat((offset2, offset3, offset4), dim=1)))
        offset_final = offset1 + offset_final
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset_final))

        return aligned_fea

#################### 135 357 579 相并联 + 时间注意力机制(k3 or k1) + target_fea (内存满了)
class TA_Fusion(nn.Module):
    def __init__(self, nf=64, center_fea=0):
        super(TA_Fusion, self).__init__()
        self.center = center_fea
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, group_fea):
        B, N, C, H, W = group_fea.size()   #[8 3 64 128 128]
        emb_0 = self.tAtt_2(group_fea[:, self.center, :, :, :].clone())  #[8 64 128 128]
        emb = self.tAtt_1(group_fea.view(-1, C, H, W)).view(B, N, -1, H, W)   #[8 3 64 128 128]

        correlation = []
        for i in range(N):  # 0,1,2
            emb_nbr = emb[:, i, :, :, :]
            correlation_temp = torch.sum(emb_nbr * emb_0, 1).unsqueeze(1)  # B, 1, H, W
            correlation.append(correlation_temp)
        correlation_pro = torch.sigmoid(torch.cat(correlation, dim=1)) # B,N,H,W
        correlation_pro = correlation_pro.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)  #[8 192 128 128]
        modulated_fea = group_fea.view(B, -1, H, W) * correlation_pro

        return modulated_fea
class MSD_1_2_TA_target(nn.Module):
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)   # 用来卷积concat后的特征
        self.mul_scale1 = nn.Conv2d(nf, nf, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变

        self.mul_scale4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale5 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale6 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变

        self.mul_scale7 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale8 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64
        self.mul_scale9 = nn.Conv2d(nf, nf, 9, 1, 4, bias=True)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.offset_conv3 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.offset_conv4 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.TAtt = TA_Fusion(nf=128, center_fea=0)

        # self.output = nn.Conv2d(4 * nf, nf, 3, 1, 1)
        self.output = nn.Conv2d(4 * nf, nf, 1, 1)
        # self.selfattention = SelfAttention(embed_dim=64 * 64),


    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.offset_conv1(offset))
        mul_scale_fea1 = self.lrelu(self.mul_scale1(offset1))
        mul_scale_fea2 = self.lrelu(self.mul_scale2(offset1))
        mul_scale_fea3 = self.lrelu(self.mul_scale3(offset1))
        offset2 = torch.cat([mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)
        offset2 = self.lrelu(self.offset_conv2(offset2))

        mul_scale_fea4 = self.lrelu(self.mul_scale4(offset1))
        mul_scale_fea5 = self.lrelu(self.mul_scale5(offset1))
        mul_scale_fea6 = self.lrelu(self.mul_scale6(offset1))
        offset3 = torch.cat([mul_scale_fea4, mul_scale_fea5, mul_scale_fea6], dim=1)
        offset3 = self.lrelu(self.offset_conv3(offset3))

        mul_scale_fea7 = self.lrelu(self.mul_scale7(offset1))
        mul_scale_fea8 = self.lrelu(self.mul_scale8(offset1))
        mul_scale_fea9 = self.lrelu(self.mul_scale9(offset1))
        offset4 = torch.cat([mul_scale_fea7, mul_scale_fea8, mul_scale_fea9], dim=1)
        offset4 = self.lrelu(self.offset_conv4(offset4))
        offset_list = []
        offset_list.append(target_fea)
        # offset_list.append(offset1)
        offset_list.append(offset2)
        offset_list.append(offset3)
        offset_list.append(offset4)
        offset_final = torch.stack(offset_list,dim=1)  # [8 5 128 64 64]
        offset_final = self.TAtt(offset_final)  # [8 5*128 64 64]
        offset_final = self.output(offset_final)
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset_final))

        return aligned_fea
##############################################################

#################### 135 357 579 相并联
class MSD_1_2(nn.Module):
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)   # 用来卷积concat后的特征
        self.mul_scale1 = nn.Conv2d(nf, nf, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变

        self.mul_scale4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale5 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale6 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变

        self.mul_scale7 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale8 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64
        self.mul_scale9 = nn.Conv2d(nf, nf, 9, 1, 4, bias=True)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.offset_conv3 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.offset_conv4 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.offset_conv5 = nn.Conv2d(3 * nf, nf, 1, 1, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # if
        #     self.selfattention = SelfAttention(embed_dim=64 * 64),
        # else
        #     self.selfattention = SelfAttention(embed_dim=160 * 160)

    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.offset_conv1(offset))
        mul_scale_fea1 = self.lrelu(self.mul_scale1(offset1))
        mul_scale_fea2 = self.lrelu(self.mul_scale2(offset1))
        mul_scale_fea3 = self.lrelu(self.mul_scale3(offset1))
        offset2 = torch.cat([mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)
        offset2 = self.offset_conv2(offset2)

        mul_scale_fea4 = self.lrelu(self.mul_scale4(offset1))
        mul_scale_fea5 = self.lrelu(self.mul_scale5(offset1))
        mul_scale_fea6 = self.lrelu(self.mul_scale6(offset1))
        offset3 = torch.cat([mul_scale_fea4, mul_scale_fea5, mul_scale_fea6], dim=1)
        offset3 = self.offset_conv3(offset3)

        mul_scale_fea7 = self.lrelu(self.mul_scale7(offset1))
        mul_scale_fea8 = self.lrelu(self.mul_scale8(offset1))
        mul_scale_fea9 = self.lrelu(self.mul_scale9(offset1))
        offset4 = torch.cat([mul_scale_fea7, mul_scale_fea8, mul_scale_fea9], dim=1)
        offset4 = self.offset_conv4(offset4)

        offset_final = self.lrelu(self.offset_conv5(torch.cat((offset2, offset3, offset4), dim=1)))
        offset_final = offset1 + offset_final
        # b, c, h, w = neibor_fea.size()
        # # self.selfattention = SelfAttention(embed_dim=h*w)
        # offset_final = offset_final.view(b, c, -1)
        # offset_final = (self.selfattention(offset_final)).view(b, c, h, w)
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset_final))

        return aligned_fea

# 还原3层尺度串联（相比于MSD1，最后一层换成1*1卷积）(加入注意力机制)
class MSD_1_1_TA(nn.Module):
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)   # 用来卷积concat后的特征
        self.mul_scale1 = nn.Conv2d(nf, nf, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale4 = nn.Conv2d(nf, nf, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # (64-7+2*3)/1+1=64
        self.mul_scale6 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset

        self.offset_conv3 = nn.Conv2d(3 * nf, nf, 1, 1, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset

        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.TAtt = TA_Fusion(nf=128, center_fea=2)
        self.output = nn.Conv2d(4 * nf, nf, 1, 1)

    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.offset_conv1(offset))
        mul_scale_fea1 = self.lrelu(self.mul_scale1(offset1))
        mul_scale_fea2 = self.lrelu(self.mul_scale2(offset1))
        mul_scale_fea3 = self.lrelu(self.mul_scale3(offset1))
        offset2 = torch.cat([mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)
        offset2 = self.lrelu(self.offset_conv2(offset2))

        mul_scale_fea4 = self.lrelu(self.mul_scale4(offset2))
        mul_scale_fea5 = self.lrelu(self.mul_scale5(offset2))
        mul_scale_fea6 = self.lrelu(self.mul_scale6(offset2))
        offset3 = torch.cat([mul_scale_fea4, mul_scale_fea5, mul_scale_fea6], dim=1)
        offset4 = self.lrelu(self.offset_conv3(offset3))

        offset_list = []
        offset_list.append(offset1)
        offset_list.append(offset2)
        offset_list.append(target_fea)
        offset_list.append(offset4)
        offset_final = torch.stack(offset_list,dim=1)  # [8 3 128 64 64]
        offset_final = self.TAtt(offset_final)  # [8 3*128 64 64]
        offset_final = self.output(offset_final)
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset_final))

        # offset_final = offset1 + offset2
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, aligned_fea))

        return aligned_fea

# 还原3层尺度串联（相比于MSD1，最后一层还原成1*1卷积）
class MSD_1_1(nn.Module):
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)   # 用来卷积concat后的特征
        self.mul_scale1 = nn.Conv2d(nf, nf, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # (64-7+2*3)/1+1=64
        self.mul_scale4 = nn.Conv2d(nf, nf, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale6 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.offset_conv3 = nn.Conv2d(3 * nf, nf, 1, 1, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset

        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.offset_conv1(offset))
        mul_scale_fea1 = self.lrelu(self.mul_scale1(offset1))
        mul_scale_fea2 = self.lrelu(self.mul_scale2(offset1))
        mul_scale_fea3 = self.lrelu(self.mul_scale3(offset1))
        offset2 = torch.cat([mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)
        offset2 = self.lrelu(self.offset_conv2(offset2))

        mul_scale_fea4 = self.lrelu(self.mul_scale4(offset2))
        mul_scale_fea5 = self.lrelu(self.mul_scale5(offset2))
        mul_scale_fea6 = self.lrelu(self.mul_scale6(offset2))
        offset3 = torch.cat([mul_scale_fea4, mul_scale_fea5, mul_scale_fea6], dim=1)
        offset4 = self.lrelu(self.offset_conv3(offset3))

        # offset_final = offset1 + offset2
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset4))

        return aligned_fea

# 3层尺度普通串联（无残差）
class MSD_1(nn.Module):
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)   # 用来卷积concat后的特征
        self.mul_scale1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64
        self.mul_scale4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale5 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale6 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.offset_conv3 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset

        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.offset_conv1(offset))
        mul_scale_fea1 = self.lrelu(self.mul_scale1(offset1))
        mul_scale_fea2 = self.lrelu(self.mul_scale2(offset1))
        mul_scale_fea3 = self.lrelu(self.mul_scale3(offset1))
        offset2 = torch.cat([mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)
        offset2 = self.lrelu(self.offset_conv2(offset2))

        mul_scale_fea4 = self.lrelu(self.mul_scale4(offset2))
        mul_scale_fea5 = self.lrelu(self.mul_scale5(offset2))
        mul_scale_fea6 = self.lrelu(self.mul_scale6(offset2))
        offset3 = torch.cat([mul_scale_fea4, mul_scale_fea5, mul_scale_fea6], dim=1)
        offset4 = self.lrelu(self.offset_conv3(offset3))

        # offset_final = offset1 + offset2
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset4))

        return aligned_fea

# MSD_yuan
class MSD(nn.Module):  # Ours
    '''
    Alignment with multi-scale deformable conv
    '''
    def __init__(self, nf=128, groups=8, dilation=1):
        super(MSD, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * nf, nf, 3, 1, 1, bias=True)   # 用来卷积concat后的特征
        self.mul_scale1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)   # 用得到的offset指导可变形操作

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, neibor_fea, target_fea):
        offset = torch.cat([neibor_fea, target_fea], dim=1)
        offset1 = self.lrelu(self.offset_conv1(offset))
        mul_scale_fea1 = self.lrelu(self.mul_scale1(offset1))
        mul_scale_fea2 = self.lrelu(self.mul_scale2(offset1))
        mul_scale_fea3 = self.lrelu(self.mul_scale3(offset1))
        offset2 = torch.cat([mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)
        offset2 = self.lrelu(self.offset_conv2(offset2))
        offset_final = offset1 + offset2
        aligned_fea = self.lrelu(self.dcnpack(neibor_fea, offset_final))

        return aligned_fea

######################  HAT  ##########################

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

# Mlp_yuan
class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 3层多尺度串联2层，且加入残差（无DW-Conv）
class Mlp_1(nn.Module):
    def __init__(self, input=180, dilation=1):
        super().__init__()
        nf = 180
        self.input = nn.Conv2d(input, nf, 3, 1, 1, bias=True)

        self.mul_scale1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64

        self.mul_scale4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale5 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale6 = nn.Conv2d(nf, nf, 7, 1, 3, bias=True)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(3 * nf, nf, 1, 1, bias=True, dilation=dilation)  # 用来聚合concat后的多尺度特征,得到的结果与conv1的结果相加，得到最终的offset
        self.offset_conv3 = nn.Conv2d(3 * nf, nf, 1, 1, bias=True, dilation=dilation)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x):
        x = self.input(x)
        mul_scale_fea1 = self.lrelu(self.mul_scale1(x))
        mul_scale_fea2 = self.lrelu(self.mul_scale2(x))
        mul_scale_fea3 = self.lrelu(self.mul_scale3(x))
        fea2 = torch.cat([mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)
        fea2 = self.lrelu(self.offset_conv2(fea2))
        mul_scale_fea4 = self.lrelu(self.mul_scale4(fea2))
        mul_scale_fea5 = self.lrelu(self.mul_scale5(fea2))
        mul_scale_fea6 = self.lrelu(self.mul_scale6(fea2))
        fea3 = torch.cat([mul_scale_fea4, mul_scale_fea5, mul_scale_fea6], dim=1)
        # fea4 = self.lrelu(self.offset_conv3(fea3))
        # fea = x + fea2 + fea4
        fea = self.lrelu(self.offset_conv3(fea3))

        return fea

# 3层多尺度串联2层，且加入残差（有DW-Conv）
class Mlp_2(nn.Module):
    def __init__(self, input=180, bias=False):
        super().__init__()
        nf = 180
        self.input = nn.Conv2d(input, nf, 3, 1, 1, bias=bias)

        self.mul_scale1 = nn.Conv2d(nf, nf, 3, 1, 1, groups=nf, bias=bias)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf, nf, 5, 1, 2, groups=nf, bias=bias)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf, 7, 1, 3, groups=nf, bias=bias)   # (64-7+2*3)/1+1=64

        self.mul_scale4 = nn.Conv2d(nf, nf, 3, 1, 1, groups=nf, bias=bias)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale5 = nn.Conv2d(nf, nf, 5, 1, 2, groups=nf, bias=bias)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale6 = nn.Conv2d(nf, nf, 7, 1, 3, groups=nf, bias=bias)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        self.offset_conv2 = nn.Conv2d(3 * nf, nf, 1, 1, bias=bias)
        self.offset_conv3 = nn.Conv2d(3 * nf, nf, 1, 1, bias=bias)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.relu = nn.ReLU()    #  可以尝试

    def forward(self, x):
        x = self.input(x)
        mul_scale_fea1 = self.lrelu(self.mul_scale1(x))
        mul_scale_fea2 = self.lrelu(self.mul_scale2(x))
        mul_scale_fea3 = self.lrelu(self.mul_scale3(x))
        fea2 = torch.cat([mul_scale_fea1, mul_scale_fea2, mul_scale_fea3], dim=1)
        fea2 = self.lrelu(self.offset_conv2(fea2))
        mul_scale_fea4 = self.lrelu(self.mul_scale4(fea2))
        mul_scale_fea5 = self.lrelu(self.mul_scale5(fea2))
        mul_scale_fea6 = self.lrelu(self.mul_scale6(fea2))
        fea3 = torch.cat([mul_scale_fea4, mul_scale_fea5, mul_scale_fea6], dim=1)
        # fea4 = self.lrelu(self.offset_conv3(fea3))
        # fea = x + fea2 + fea4
        fea = self.lrelu(self.offset_conv3(fea3))
        return fea

# 3层多尺度串联2层，且加入残差（有DW-Conv）,还原MFL
class Mlp_3(nn.Module):
    def __init__(self, input=180, bias=False):
        super().__init__()
        nf = 180
        self.input = nn.Conv2d(input, nf, 3, 1, 1, bias=bias)

        self.mul_scale1 = nn.Conv2d(nf, nf, 3, 1, 1, groups=nf, bias=bias)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale2 = nn.Conv2d(nf, nf, 5, 1, 2, groups=nf, bias=bias)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale3 = nn.Conv2d(nf, nf, 7, 1, 3, groups=nf, bias=bias)   # (64-7+2*3)/1+1=64

        self.mul_scale4 = nn.Conv2d(nf, nf, 3, 1, 1, groups=nf, bias=bias)   # 第一个多尺度，3*3，（64-3+2*1）/1+1=64，尺寸不变
        self.mul_scale5 = nn.Conv2d(nf, nf, 5, 1, 2, groups=nf, bias=bias)   # 提取5*5尺度特征，（64-5+2*2）/1+1=64，尺寸不变
        self.mul_scale6 = nn.Conv2d(nf, nf, 7, 1, 3, groups=nf, bias=bias)   # (64-7+2*3)/1+1=64
        # k'=d*(k-1)+1,k=3
        # d=1,k'=3,stride=1,padding=1  d=2,k'=5,s=1,p=2,  d=3,k'=7,s=1,p=3
        # self.offset_conv2 = nn.Conv2d(3 * nf, nf, 1, 1, bias=bias)
        self.offset_conv3 = nn.Conv2d(3 * nf, nf, 1, 1, bias=bias)

        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input(x)

        fea11, fea12, fea13 = self.relu(self.mul_scale1(x)).split(60, dim=1)
        fea21, fea22, fea23 = self.relu(self.mul_scale2(x)).split(60, dim=1)
        fea31, fea32, fea33 = self.relu(self.mul_scale3(x)).split(60, dim=1)

        fea1 = torch.cat([fea11, fea21, fea31], dim=1)
        fea2 = torch.cat([fea12, fea22, fea32], dim=1)
        fea3 = torch.cat([fea13, fea23, fea33], dim=1)

        fea4 = self.relu(self.mul_scale4(fea1))
        fea5 = self.relu(self.mul_scale5(fea2))
        fea6 = self.relu(self.mul_scale6(fea3))

        fea = torch.cat([fea4, fea5, fea6], dim=1)
        fea = self.offset_conv3(fea)
        return fea

def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

####################### HAB_ours #################
class HAB(nn.Module):
    r""" Hybrid Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim=180,
                 input_resolution=(64,64),
                 num_heads=6,
                 window_size=16,
                 shift_size=0,
                 conv_scale=0.01,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        # self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        # self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.conv_scale = conv_scale
        # self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.swin = SwinPCCA(c1=dim, c2=dim, e=1)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def calculate_rpi_sa(self):
        # calculate relative position index for SA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    # def calculate_rpi_oca(self):
    #     # calculate relative position index for OCA
    #     window_size_ori = self.window_size
    #     window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)
    #
    #     coords_h = torch.arange(window_size_ori)
    #     coords_w = torch.arange(window_size_ori)
    #     coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
    #     coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws
    #
    #     coords_h = torch.arange(window_size_ext)
    #     coords_w = torch.arange(window_size_ext)
    #     coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
    #     coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse
    #
    #     relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]  # 2, ws*ws, wse*wse
    #
    #     relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
    #     relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
    #     relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1
    #
    #     relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
    #     relative_position_index = relative_coords.sum(-1)
    #     return relative_position_index

    def forward(self, x):

        b,c,h,w = x.size()

        # x_size = (128,128)
        x_size = (h,w)

        relative_position_index_SA = self.calculate_rpi_sa()
        # relative_position_index_OCA = self.calculate_rpi_oca()
        # self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        # self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)
        attn_mask = self.calculate_mask(x_size).to(x.device)

        x1 = x.permute(0,2,3,1)
        x = x1.view(b, h * w, c)
        shortcut = x

        x = self.norm1(x)     # [1 16384 180]
        x = x.view(b, h, w, c)   # [1 128 128 180]

        # Conv_X
        # conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = self.swin(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # [1 16384 180]

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows        # 图像切割
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        # attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)
        attn_windows = self.attn(x_windows, rpi=relative_position_index_SA, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c  [1 128 128 180]

        # reverse cyclic shift   循环平移
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)    # [1 16384 180]

        # FFN
        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale   # [1 16384 180]
        # x = x + self.drop_path(self.mlp(self.norm2(x)))   # [1 16384 180]

        x2 = x.view(b, h, w, c)
        x = x2.permute(0,3,1,2)
        return x

####################### Standard_Swin_transformer #################
# class HAB(nn.Module):
#     r""" Hybrid Attention Block.
#
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         num_heads (int): Number of attention heads.
#         window_size (int): Window size.
#         shift_size (int): Shift size for SW-MSA.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """
#
#     def __init__(self,
#                  dim=180,
#                  input_resolution=(64,64),
#                  num_heads=6,
#                  window_size=16,
#                  shift_size=0,
#                  conv_scale=0.01,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop=0.,
#                  attn_drop=0.,
#                  drop_path=0.,
#                  norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         # self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         # self.mlp_ratio = mlp_ratio
#         if min(self.input_resolution) <= self.window_size:
#             # if window size is larger than input resolution, we don't partition windows
#             self.shift_size = 0
#             self.window_size = min(self.input_resolution)
#         assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
#
#         self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention(
#             dim,
#             window_size=to_2tuple(self.window_size),
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop)
#
#         self.conv_scale = conv_scale
#         # self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         # self.norm2 = norm_layer(dim)
#         # mlp_hidden_dim = int(dim * mlp_ratio)
#         # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         self.swin = SwinPCCA(c1=dim, c2=dim, e=1)
#
#     def calculate_mask(self, x_size):
#         # calculate attention mask for SW-MSA
#         h, w = x_size
#         img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
#         h_slices = (slice(0, -self.window_size), slice(-self.window_size,
#                                                        -self.shift_size), slice(-self.shift_size, None))
#         w_slices = (slice(0, -self.window_size), slice(-self.window_size,
#                                                        -self.shift_size), slice(-self.shift_size, None))
#         cnt = 0
#         for h in h_slices:
#             for w in w_slices:
#                 img_mask[:, h, w, :] = cnt
#                 cnt += 1
#
#         mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
#         mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#         attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#         attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#
#         return attn_mask
#
#     def calculate_rpi_sa(self):
#         # calculate relative position index for SA
#         coords_h = torch.arange(self.window_size)
#         coords_w = torch.arange(self.window_size)
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         return relative_position_index
#
#     # def calculate_rpi_oca(self):
#     #     # calculate relative position index for OCA
#     #     window_size_ori = self.window_size
#     #     window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)
#     #
#     #     coords_h = torch.arange(window_size_ori)
#     #     coords_w = torch.arange(window_size_ori)
#     #     coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
#     #     coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws
#     #
#     #     coords_h = torch.arange(window_size_ext)
#     #     coords_w = torch.arange(window_size_ext)
#     #     coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
#     #     coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse
#     #
#     #     relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]  # 2, ws*ws, wse*wse
#     #
#     #     relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
#     #     relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
#     #     relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1
#     #
#     #     relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
#     #     relative_position_index = relative_coords.sum(-1)
#     #     return relative_position_index
#
#     def forward(self, x):
#
#         b,c,h,w = x.size()
#
#         # x_size = (128,128)
#         x_size = (h,w)
#
#         relative_position_index_SA = self.calculate_rpi_sa()
#         # relative_position_index_OCA = self.calculate_rpi_oca()
#         # self.register_buffer('relative_position_index_SA', relative_position_index_SA)
#         # self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)
#         attn_mask = self.calculate_mask(x_size).to(x.device)
#
#         x1 = x.permute(0,2,3,1)
#         x = x1.view(b, h * w, c)
#         shortcut = x
#
#         x = self.norm1(x)     # [1 16384 180]
#         x = x.view(b, h, w, c)   # [1 128 128 180]
#
#         # Conv_X
#         # conv_x = self.conv_block(x.permute(0, 3, 1, 2))
#         # conv_x = self.swin(x.permute(0, 3, 1, 2))
#         # conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # [1 16384 180]
#
#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#             attn_mask = attn_mask
#         else:
#             shifted_x = x
#             attn_mask = None
#
#         # partition windows        # 图像切割
#         x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
#         x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c
#
#         # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
#         # attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)
#         attn_windows = self.attn(x_windows, rpi=relative_position_index_SA, mask=attn_mask)
#
#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
#         shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c  [1 128 128 180]
#
#         # reverse cyclic shift   循环平移
#         if self.shift_size > 0:
#             attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             attn_x = shifted_x
#         attn_x = attn_x.view(b, h * w, c)    # [1 16384 180]
#
#         # FFN
#         x = shortcut + self.drop_path(attn_x) # + conv_x * self.conv_scale   # [1 16384 180]
#         # x = x + self.drop_path(self.mlp(self.norm2(x)))   # [1 16384 180]
#
#         x2 = x.view(b, h, w, c)
#         x = x2.permute(0,3,1,2)
#         return x

####################### XiaoYi_BSVSR( + Optical Flow) #################
class DeformableAttnBlock_FUSION(nn.Module):
    def __init__(self, n_heads=4, n_levels=3, n_points=4, d_model=32):
        super().__init__()
        self.n_levels = n_levels

        self.defor_attn = MSDeformAttn_Fusion(d_model=d_model, n_levels=3, n_heads=n_heads, n_points=n_points)
        self.feed_forward = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.emb_qk = nn.Conv2d(3 * d_model + 4, 3 * d_model, kernel_size=3, padding=1)
        self.emb_v = nn.Conv2d(3 * d_model, 3 * d_model, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.feedforward = nn.Sequential(
            nn.Conv2d(2 * d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.fusion = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def preprocess(self, srcs):
        bs, t, c, h, w = srcs.shape
        masks = [torch.zeros((bs, h, w)).bool().to(srcs.device) for _ in range(t)]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lv1 in range(t):
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs.device)
        return spatial_shapes, valid_ratios

    def forward(self, frame, srcframe, flow_forward, flow_backward):
        b, t, c, h, w = frame.shape
        # bs,t,c,h,w = frame.shape
        warp_fea01 = warp(frame[:, 0], flow_backward[:, 0])
        warp_fea21 = warp(frame[:, 2], flow_forward[:, 1])

        qureys = self.act(self.emb_qk(
            torch.cat([warp_fea01, frame[:, 1], warp_fea21, flow_forward[:, 1], flow_backward[:, 0]], 1))).reshape(b, t,
                                                                                                                   c, h,
                                                                                                                   w)

        value = self.act(self.emb_v(frame.reshape(b, t * c, h, w)).reshape(b, t, c, h, w))

        spatial_shapes, valid_ratios = self.preprocess(value)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self.get_reference_points(spatial_shapes[0].reshape(1, 2), valid_ratios, device=value.device)

        output = self.defor_attn(qureys, reference_points, value, spatial_shapes, level_start_index, None, flow_forward,
                                 flow_backward)

        output = self.feed_forward(output)
        output = output.reshape(b, c, h, w) + frame[:, 1]

        tseq_encoder_0 = torch.cat([output, srcframe[:, 1]], 1)
        output = output.reshape(b, c, h, w) + self.feedforward(tseq_encoder_0)
        output = self.fusion(output)
        return output

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).reshape(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).reshape(-1, 1).repeat(1, W)
    xx = xx.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    grid = grid.to(x.device)
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, padding_mode='border', align_corners=True)
    # mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    # mask = nn.functional.grid_sample(mask, vgrid,align_corners=True )

    # mask[mask < 0.999] = 0
    # mask[mask > 0] = 1

    # output = output * mask

    return output

####################### HAB_yuan_HAT #################


# HAB_Mlp_x
class HAB_1(nn.Module):
    r""" Hybrid Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim=180,
                 input_resolution=(64,64),
                 num_heads=6,
                 window_size=16,
                 shift_size=0,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.conv_scale = conv_scale
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm(dim=dim, LayerNorm_type='WithBias')
        self.mlp = Mlp()  # Mlp_1  Mlp_2

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def calculate_rpi_sa(self):
        # calculate relative position index for SA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def forward(self, x):

        b,c,h,w = x.size()

        # x_size = (128,128)
        x_size = (h,w)

        relative_position_index_SA = self.calculate_rpi_sa()
        # relative_position_index_OCA = self.calculate_rpi_oca()
        # self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        # self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)
        attn_mask = self.calculate_mask(x_size).to(x.device)

        x1 = x.permute(0,2,3,1)
        x = x1.view(b, h * w, c)
        shortcut = x

        x = self.norm1(x)     # [1 16384 180]
        x = x.view(b, h, w, c)   # [1 128 128 180]

        # Conv_X
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))  # [1 180 128 128]
        # conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # [1 16384 180]
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h, w, c)  # [1 16384 180]   # Mlp_1  Mlp_2

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows        # 图像切割
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        # attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)
        attn_windows = self.attn(x_windows, rpi=relative_position_index_SA, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c  [1 128 128 180]

        # reverse cyclic shift   循环平移
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)    # [1 16384 180]

    ###   Mlp_1  Mlp_2
        shortcut = shortcut.view(b, h, w, c)
        attn_x = self.drop_path(attn_x)
        attn_x = attn_x.view(b, h, w, c)
        x = shortcut + attn_x + conv_x * self.conv_scale
        x = x.reshape(b, c, h, w)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
    ###

        return x

############################# TDAN ############################
# from torch.nn.modules.module import Module
# from torch.nn.modules.utils import _pair
# from model.functions import conv_offset2d
# class ConvOffset2d(Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  num_deformable_groups=1):
#         super(ConvOffset2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)
#         self.num_deformable_groups = num_deformable_groups
#
#         self.weight = nn.Parameter(
#             torch.Tensor(out_channels, in_channels, *self.kernel_size))
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, offset):
#         return conv_offset2d(input, offset, self.weight, self.stride,
#                              self.padding, self.dilation,
#                              self.num_deformable_groups)
#
class Res_Block(nn.Module):
    def __init__(self):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res
####################################################################

#################### Mlp_x 的 LayerNorm #########################
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weigh

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
###############################################################