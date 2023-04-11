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
                up_sample_factor: int=2, 
                dilation: int=1, 
                bias: bool=True,
                groups: int=1):
        super(DeformConvTranspose, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_sample_factor = up_sample_factor
        self.dilation = (dilation, dilation)
        self.kernel_size = (up_sample_factor * 2, up_sample_factor * 2)
        self.stride = up_sample_factor
        self.padding = up_sample_factor // 2
        self.groups = groups
        channels_ = self.groups * 3 * self.kernel_size[0] * self.kernel_size[1]

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups,
                                   self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          groups=groups,
                                          bias=True)
        self.dcn = deform_conv2d
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

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

    def forward(self, input: torch.Tensor):
        input = self.input_transform(input)
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.softmax(mask, dim=1)
        return self.dcn(input=input, 
                        offset=offset, 
                        weight=self.weight.flip(-1, -2), 
                        bias=self.bias, 
                        dilation=self.dilation, 
                        mask=mask)

if __name__ == '__main__':
    a = torch.rand(1, 1, 7, 7)
    #b = input_transform(a)
    conv = DeformConvTranspose(1, 1)
    c = conv(a)
    print(c.shape)