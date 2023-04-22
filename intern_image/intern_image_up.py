import torch
from torch import nn 
from timm.models.layers import trunc_normal_
import ops_dcnv3.modules as opsm
from .intern_image import InternImage, InternImageBlock, build_norm_layer, MLPLayer, to_channels_first, to_channels_last


class InternImageUp(nn.Module):

    def __init__(self,
                 uplevel,
                 core_op='DCNv3',
                 coreup_op='DCNv3Up',
                 channels=64,
                 depths=[3, 4, 18, 5],
                 groups=[3, 6, 12, 24],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 offset_scale=1.0,
                 layer_scale=None,
                 with_cp=False,
                 dw_kernel_size=None, 
                 post_norm_block_ids=None, 
                 res_post_norm=False, 
                 center_feature_scale=False):
        super().__init__()
        self.uplevel = uplevel
        self.core_op=core_op
        self.coreup_op=coreup_op
        self.channels = channels
        self.depths = depths
        self.groups = groups
        self.post_norm = post_norm
        self.center_feature_scale = center_feature_scale
        self.mlp_ratio = mlp_ratio
        self.num_levels = len(depths)

        for i in range(self.num_levels - uplevel):
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:i+1]) + 
                                                   (self.num_levels - i - 1) * depths[0])]
            if drop_path_type == 'uniform':
                for i in range(len(dpr)):
                    dpr[i] = drop_path_rate
            _channels = int(channels * 2**i)
            _level = InternImageBlock(
                core_op=getattr(opsm, core_op),
                channels=_channels,
                depth=depths[0],
                groups=groups[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i+1]):sum(depths[:i+1]) + uplevel * depths[0]],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                downsample=False,
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                post_norm_block_ids=post_norm_block_ids, # for InternImage-H/G
                res_post_norm=res_post_norm, # for InternImage-H/G
                center_feature_scale=center_feature_scale # for InternImage-H/G
            )
            level = nn.Sequential(to_channels_last(), _level, to_channels_first())
            levelup = nn.Sequential(
                to_channels_last(),
                getattr(opsm, coreup_op)(_channels * 2, 
                                         group=groups[i+1], 
                                         offset_scale=offset_scale,
                                         act_layer=act_layer,
                                         norm_layer=norm_layer,
                                         center_feature_scale=center_feature_scale),
                build_norm_layer(_channels * 2, norm_layer),
                MLPLayer(in_features=_channels * 2, 
                         hidden_features=_channels * 4,
                         out_features=_channels, 
                         act_layer=act_layer, 
                         drop=drop_rate),
                to_channels_first())

            self.__setattr__(f'level_{i}', level)
            self.__setattr__(f'levelup_{i}', levelup)

        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_deform_weights(self, m):
        if isinstance(m, getattr(opsm, self.core_op)) or isinstance(m, getattr(opsm, self.coreup_op)):
            m._reset_parameters()

    def forward(self, inputs):
        outputs=[]
        for i in range(self.num_levels - self.uplevel):
            x = self.__getattr__(f'level_{i}')(inputs[i])
            x_up = self.__getattr__(f'levelup_{i}')(inputs[i+1])
            outputs.append(x + x_up)

        return outputs


if __name__ == '__main__':
    a = torch.randn((2, 3, 960, 544), dtype=torch.float32, device='cuda:0')
    pm = InternImage(groups=[4, 8, 16, 32]).to('cuda:0')
    a = pm(a)
    _423 = InternImageUp(uplevel=1, groups=[4, 8, 16, 32]).to('cuda:0')
    _322 = InternImageUp(uplevel=2, groups=[4, 8, 16, 32]).to('cuda:0')
    _221 = InternImageUp(uplevel=3, groups=[4, 8, 16, 32]).to('cuda:0')
    c = _423(a)
    c = _322(c)
    c = _221(c)

    d = c