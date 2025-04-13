import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F
# import src.net.attention as att
import functools
import torch.nn as nn

# from models._common import Attention, AttentionLePE
from models.seaformer import Sea_Attention
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2

from models.vision_transformer import SwinUnet

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, window_size, relative_pos_embedding=True):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2)) if not relative_pos_embedding else None
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: t.view(b, n_h, n_w, h, -1), qkv)

        dots = torch.einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = out.view(b, n_h, n_w, -1)
        out = self.to_out(out)

        return out


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)
        self.downscaling_factor = downscaling_factor

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x)
        x = x.view(b, -1, new_h, new_w)
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class ViTs(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size, relative_pos_embedding=True):
        super().__init__()

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension, downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(WindowAttention(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, window_size=window_size, relative_pos_embedding=relative_pos_embedding))

        self.channel_att = ChannelGate(hidden_dimension, reduction_ratio=16, pool_types=['avg', 'max'])

    def forward(self, x):
        x = self.patch_partition(x)

        for layer in self.layers:
            x = layer(x)

        x = self.channel_att(x)
        return x


class PFANX(nn.Module):
    def __init__(self, *, input_nc, output_nc, ngf, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=8,
                 downscaling_factors=(1, 1, 1, 1), relative_pos_embedding=True, norm_layer_1='batch'):
        super().__init__()
        from models.networks import Block, BlockV2

        if type(norm_layer_1) == functools.partial:
            use_bias = norm_layer_1.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer_1 == nn.InstanceNorm2d

        model_1 = [
            nn.Conv2d(input_nc, ngf, 1, 1, 0),  # [3, 256, 256] -> [64, 256, 256]
            norm_layer_1(ngf),  # [64, 256, 256]
            nn.LeakyReLU(0.05)]  # [64, 256, 256]

        model_3 = []
        model_3_1 = [nn.Tanh()]
        model_3 += [nn.Conv2d(hidden_dim, output_nc, 1, 1, 0)]

        self.model_1 = nn.Sequential(*model_1)

        self.convnext1 = Block(ngf)
        self.convnext2 = Block(ngf)

        # self.vit = ViTs(in_channels=hidden_dim, hidden_dimension=hidden_dim, layers=layers[2],
        #                 downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
        #                 window_size=4, relative_pos_embedding=relative_pos_embedding)

        self.vit = ViTs(in_channels=hidden_dim, hidden_dimension=hidden_dim, layers=layers[2],
                        downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                        window_size=4, relative_pos_embedding=False)

        # self.vit = SwinUnet(img_size=256, patch_size=4, window_size=8, in_chans=hidden_dim, num_classes=hidden_dim, embed_dim=hidden_dim)
        self.out_layer_conv = nn.Conv2d(hidden_dim, output_nc, 1, 1, 0)

        self.model_3_1 = nn.Sequential(*model_3_1)

    def forward(self, img):
        x = self.model_1(img)

        x1 = self.convnext1(x)
        x1 = self.convnext2(x1)

        x2 = self.vit(x1) + x

        x3 = self.out_layer_conv(x2)
        x3 = self.model_3_1(x3)

        return x3


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_feature_map(img_batch):
    feature_map = img_batch.cpu()
    print(feature_map.shape)

    feature_map_combination = []
    # plt.figure()

    num_pic = feature_map.shape[1]
    # row, col = get_row_col(num_pic)
    row, col = 8, 8
    num_pic = min(num_pic, 9)
    for i in range(0, 64):
        feature_map_split = feature_map[0, i, :, :]
        feature_map_combination.append(feature_map_split)
        # plt.subplot(row, col, i + 1)
        # plt.imshow(feature_map_split)
        # plt.axis('off')

    # plt.savefig('feature_map.png')     plt.show()
    # plt.show()
    # ??????1?1 ??

    feature_map_sum = np.sum(ele for ele in feature_map_combination)
    return feature_map_sum


def swin_t(hidden_dim=64, layers=(2, 2, 2, 2), heads=(4, 4, 4, 4), **kwargs):
    return PFANX(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_s(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return PFANX(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_b(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return PFANX(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_l(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return PFANX(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)
