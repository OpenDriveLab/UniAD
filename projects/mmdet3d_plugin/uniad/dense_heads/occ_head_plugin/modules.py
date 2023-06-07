#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
from .utils import calculate_birds_eye_view_parameters
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule, build_conv_layer
from einops import rearrange
from collections import OrderedDict

# Grid sampler
# Sample a smaller receptive-field bev from larger one
class BevFeatureSlicer(nn.Module):
    def __init__(self, grid_conf, map_grid_conf):
        super().__init__()
        if grid_conf == map_grid_conf:
            self.identity_mapping = True
        else:
            self.identity_mapping = False

            bev_resolution, bev_start_position, bev_dimension= calculate_birds_eye_view_parameters(
                grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound']
            )

            map_bev_resolution, map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
                map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound']
            )

            self.map_x = torch.arange(
                map_bev_start_position[0], map_grid_conf['xbound'][1], map_bev_resolution[0])

            self.map_y = torch.arange(
                map_bev_start_position[1], map_grid_conf['ybound'][1], map_bev_resolution[1])

            # convert to normalized coords
            self.norm_map_x = self.map_x / (- bev_start_position[0])
            self.norm_map_y = self.map_y / (- bev_start_position[1])

            tmp_m, tmp_n = torch.meshgrid(
                self.norm_map_x, self.norm_map_y)  # indexing 'ij'
            tmp_m, tmp_n = tmp_m.T, tmp_n.T  # change it to the 'xy' mode results

            self.map_grid = torch.stack([tmp_m, tmp_n], dim=2)

    def forward(self, x):
        # x: bev feature map tensor of shape (b, c, h, w)
        if self.identity_mapping:
            return x
        else:
            grid = self.map_grid.unsqueeze(0).type_as(
                x).repeat(x.shape[0], 1, 1, 1)  # (b, h, w, 2)

            return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)

# General layers
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SimpleConv2d(BaseModule):
    def __init__(self, in_channels, 
                       out_channels, 
                       
                       conv_channels=64,
                       num_conv=1,
                       conv_cfg=dict(type='Conv2d'),
                       norm_cfg=dict(type='BN2d'),
                       bias='auto',
                       init_cfg=None,
                       ):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)
        self.out_channels = out_channels
        if num_conv == 1:
            conv_channels = in_channels

        conv_layers = []
        c_in = in_channels
        for i in range(num_conv-1):
            conv_layers.append(
                ConvModule(
                    c_in,
                    conv_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                )
            )
            c_in = conv_channels
        # No norm and relu in last conv
        conv_layers.append(
            build_conv_layer(
                conv_cfg,
                conv_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
        )
        self.conv_layers = nn.Sequential(*conv_layers)

        if init_cfg is None:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x):
        b, c_in, h_in, w_in = x.size()
        out = self.conv_layers(x)
        assert out.size() == (b, self.out_channels, h_in, w_in)  # sanity check
        return out

# Decoder
class CVT_DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor, upsample, with_relu=True):
        super().__init__()

        dim = out_channels // factor

        if upsample:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None
        
        self.with_relu = with_relu
        if self.with_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up
        if self.with_relu:
            return self.relu(x)
        return x

class CVT_Decoder(BaseModule):
    def __init__(self, dim, blocks, residual=True, factor=2, upsample=True, use_checkpoint=False, init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)

        layers = []
        channels = dim

        for i, out_channels in enumerate(blocks):
            with_relu = i < len(blocks) - 1  # if not last block, with relu
            layer = CVT_DecoderBlock(channels, out_channels, dim, residual, factor, upsample, with_relu=with_relu)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels
        self.use_checkpoint = use_checkpoint
        
        if init_cfg is None:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x):
        b, t = x.size(0), x.size(1)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        y = x
        for layer in self.layers:
            if self.use_checkpoint:
                y = checkpoint(layer, y, x)
            else:
                y = layer(y, x)
        
        y = rearrange(y, '(b t) c h w -> b t c h w', b=b, t=t)
        return y


# Conv modules
class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip

class Interpolate(nn.Module):
    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self._interpolate = nn.functional.interpolate
        self._scale_factor = scale_factor

    def forward(self, x):
        return self._interpolate(x, scale_factor=self._scale_factor, mode='bilinear', align_corners=False)

class Bottleneck(nn.Module):
    """
    Defines a bottleneck module with a residual connection
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        dilation=1,
        groups=1,
        upsample=False,
        downsample=False,
        dropout=0.0,
    ):
        super().__init__()
        self._downsample = downsample
        bottleneck_channels = int(in_channels / 2)
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        # Define the main conv operation
        assert dilation == 1
        if upsample:
            assert not downsample, 'downsample and upsample not possible simultaneously.'
            bottleneck_conv = nn.ConvTranspose2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=1,
                stride=2,
                output_padding=padding_size,
                padding=padding_size,
                groups=groups,
            )
        elif downsample:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                stride=2,
                padding=padding_size,
                groups=groups,
            )
        else:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                padding=padding_size,
                groups=groups,
            )

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    # First projection with 1x1 kernel
                    ('conv_down_project', nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)),
                    ('abn_down_project', nn.Sequential(nn.BatchNorm2d(bottleneck_channels),
                                                       nn.ReLU(inplace=True))),
                    # Second conv block
                    ('conv', bottleneck_conv),
                    ('abn', nn.Sequential(nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True))),
                    # Final projection with 1x1 kernel
                    ('conv_up_project', nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)),
                    ('abn_up_project', nn.Sequential(nn.BatchNorm2d(out_channels),
                                                     nn.ReLU(inplace=True))),
                    # Regulariser
                    ('dropout', nn.Dropout2d(p=dropout)),
                ]
            )
        )

        if out_channels == in_channels and not downsample and not upsample:
            self.projection = None
        else:
            projection = OrderedDict()
            if upsample:
                projection.update({'upsample_skip_proj': Interpolate(scale_factor=2)})
            elif downsample:
                projection.update({'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
            projection.update(
                {
                    'conv_skip_proj': nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    'bn_skip_proj': nn.BatchNorm2d(out_channels),
                }
            )
            self.projection = nn.Sequential(projection)

    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                # pad h/w dimensions if they are odd to prevent shape mismatch with residual layer
                x = nn.functional.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
            return x_residual + self.projection(x)
        return x_residual + x
