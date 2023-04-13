from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from ..functions import DCNv3Function
from .dcnv3 import build_act_layer, build_norm_layer, _is_power_of_2, CenterFeatureScaleModule


class UpPreTransformation(nn.Module):
    def __init__(self, up_factor=2):
        super(UpPreTransformation, self).__init__()

        if up_factor % 2 != 0:
            raise ValueError(f'up_factor must be an even, but got {up_factor}')

        needed_kernel_size = up_factor * 2
        needed_padding = up_factor // 2
        self.up_factor = up_factor
        self.stride = up_factor
        self.padding = needed_kernel_size - needed_padding - 1
        self.internal_padding = self.stride - 1

    def forward(self, input):

        input = input.permute(0, 3, 1, 2)
        b, c, h, w = input.shape

        result = torch.zeros(b, c, h + self.internal_padding * (h - 1) + self.padding * 2, 
                             w + self.internal_padding * (w - 1) + self.padding * 2, 
                             device=input.device)
        H, W = result.shape[-2:]
        result[..., self.padding:H - self.padding:self.stride, 
               self.padding:W - self.padding:self.stride] = input

        return result.permute(0, 2, 3, 1)

    def __repr__(self):
        s = (
            f"{self.__class__.__name__}("
            f"internal_padding={self.internal_padding}"
            f", padding={self.padding}"
        )
        s += ")"

        return s


class DCNv3_Up(nn.Module):
    def __init__(
            self,
            channels=64,
            up_factor=2, 
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False):
        """
        DCNv3 Module
        :param channels
        :param up_factor
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.up_factor = up_factor
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = up_factor * 2
        self.dw_kernel_size = up_factor * 2
        self.dilation = 1
        self.stride = 1
        self.pad = 0
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=self.dw_kernel_size,
                stride=self.stride,
                padding=self.pad,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.input_transformation = UpPreTransformation(up_factor)
        self.offset = nn.Linear(
            channels,
            group * self.kernel_size * self.kernel_size * 2)
        self.mask = nn.Linear(
            channels,
            group * self.kernel_size * self.kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        input = self.input_transformation(input)

        x = self.input_proj(input)
        dtype = x.dtype

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)
        x_proj = x1
        N, H, W, _ = x1.shape
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1).type(dtype)

        x = DCNv3Function.apply(
            x.flip (1, 2), offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256).flip(1, 2)
        
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x