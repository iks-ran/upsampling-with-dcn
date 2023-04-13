from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn 
from torchvision.ops.deform_conv import deform_conv2d
import math

class DeformConvTranspose(nn.Module):

    def __init__(self, 
                in_channels: int, 
                out_channels: int,
                up_factor: int=2, 
                bias: bool=True):
        super(DeformConvTranspose, self).__init__()

        if up_factor % 2 != 0:
            raise ValueError(f'up_factor must be an even, but got {up_factor}')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_factor = up_factor
        self.kernel_size = (up_factor * 2, up_factor * 2)
        self.stride = up_factor
        self.padding = up_factor // 2

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels,
                                   self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.dcn = deform_conv2d
        self.reset_parameters()

    def reset_parameters(self) -> None:

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, 
                input, 
                offset, 
                mask):

        return self.dcn(input=input, 
                        offset=offset, 
                        weight=self.weight.flip(-1, -2), 
                        bias=self.bias, 
                        mask=mask)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"{self.in_channels}"
            f", {self.out_channels}"
        )
        s += f", up_factor={self.up_factor}" if self.up_factor != 2 else ""
        s += ", bias=False" if not self.bias is None else ""
        s += ")"

        return s

class DCNTranspose(nn.Module):

    def __init__(self, 
                in_channels: int, 
                out_channels: int,
                up_factor: int=2, 
                bias: bool=True,
                groups: int=1):

        super(DCNTranspose, self).__init__()

        if in_channels % groups != 0:
            raise ValueError(f'in_channels must be divisible by groups, but got {in_channels} and {groups}')
        if out_channels % groups != 0:
            raise ValueError(f'out_channels must be divisible by groups, but got {out_channels} and {groups}')
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_factor = up_factor
        self.kernel_size = (up_factor * 2, up_factor * 2)
        self.stride = up_factor
        self.padding = up_factor // 2
        self.groups = groups
        self.with_bias = bias
        channels_ = self.groups * 3 * self.kernel_size[0] * self.kernel_size[1]

        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          groups=groups,
                                          bias=bias)
        self.dcn = DeformConvTranspose(in_channels // groups, out_channels, 
                                       up_factor, bias)

        self.init_offset()

    def init_offset(self) -> None:

        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def input_transform(self, input):

        b, c, h, w = input.shape

        ip = self.stride - 1
        ph = self.kernel_size[0] - self.padding - 1
        pw = self.kernel_size[1] - self.padding - 1

        result = torch.zeros(b, c, h + ip * (h - 1) + ph * 2, 
                             w + ip * (w - 1) + pw * 2, device=input.device)
        H, W = result.shape[-2:]
        result[..., ph:H - ph:self.stride, pw:W - pw:self.stride] = input

        return result

    def forward(self, input):

        input = self.input_transform(input)

        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.softmax(mask, dim=1)

        return self.dcn(input=input, offset=offset, mask=mask)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"{self.in_channels}"
            f", {self.out_channels}"
        )
        s += f", up_factor={self.up_factor}" if self.up_factor != 2 else ""
        s += ", bias=False" if not self.with_bias else ""
        s += f", groups={self.groups}" if self.groups != 1 else ""
        s += ")"

        return s

if __name__ == '__main__':
    a = torch.rand(1, 32, 7, 7)
    #b = input_transform(a)
    conv = DCNTranspose(32, 64, up_factor=4, groups=2)
    c = conv(a)
    print(c.shape)
    print(conv)
    print(conv.dcn)