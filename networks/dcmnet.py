import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import warnings
import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

norm_cfg = dict(type='BN', requires_grad=True)


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs



class DCMNet(nn.Module):

    def __init__(self, pool_scales=(1, 2, 3, 6),
                in_channels=[96, 192, 384, 768],
                in_index=[0, 1, 2, 3],
                channels=512,
                dropout_ratio=0.1,
                num_classes=1,
                norm_cfg=norm_cfg,
                align_corners=False):

        super(DCMNet, self).__init__()
        self.in_channels = in_channels
        self.in_index = in_index
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        self.align_corners = align_corners
        self.conv_cfg = None
        self.act_cfg=dict(type='ReLU')


        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)

        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)


        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)


        self.fpn_bottleneck = nn.ModuleDict()
        self.last_layer = nn.ModuleDict()

        for scale in range(4):
            self.fpn_bottleneck["%d"%scale] = ConvModule(
                (len(self.in_channels)-scale) * self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)


            self.last_layer["%d"%scale] = nn.Conv2d(self.channels, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def psp_forward(self, inputs):
        """Forward function of PSP module."""


        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))


        psp_outs = torch.cat(psp_outs, dim=1)

        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        self.outputs = {}


        inputs = [inputs[i] for i in self.in_index]


        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for scale in range(3, -1, -1):

            temp_fpn_outs = fpn_outs[scale:]

            used_backbone_levels = len(temp_fpn_outs)
            h, w = temp_fpn_outs[0].shape[2:]
            h, w = 2*h, 2*w

            for i in range(used_backbone_levels - 1, -1, -1):
                temp_fpn_outs[i] = resize(
                    temp_fpn_outs[i],
                    size=(h, w),
                    mode='bilinear',
                    align_corners=self.align_corners)

            temp_fpn_outs = torch.cat(temp_fpn_outs, dim=1)
            output = self.fpn_bottleneck["%d"%scale](temp_fpn_outs)
            output = self.sigmoid(self.last_layer["%d"%scale](output))

            self.outputs[("disp", scale)] = output


        return self.outputs

