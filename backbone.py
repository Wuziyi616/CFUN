"""
Faster U-net

Backbone nets architectures: Pseudo-3D Residual Networks (P3D Resnet, 2017)

A shallower implementation which only contains two stages.

"""

import torch.nn as nn
import math


def conv_S(in_planes, out_planes, stride=1, padding=1):
    """conv_S is the spatial conv layer"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3),
                     stride=stride, padding=padding)


def conv_T(in_planes, out_planes, stride=1, padding=1):
    """conv_T is the temporal conv layer"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1),
                     stride=stride, padding=padding)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, block, expand=False, stride=1, ST_structure=('A', 'B', 'C')):
        """A wrapper for different Bottlenecks.
        block: identify Block_A/B/C.
        expand: whether to expand the final output channel by multiplying expansion.
        """
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.expand = expand

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)

        self.ST = list(ST_structure)[(block - 1) % len(ST_structure)]
        self.conv2 = conv_S(planes, planes, stride=1, padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv_T(planes, planes, stride=1, padding=(1, 0, 0))
        self.bn3 = nn.BatchNorm3d(planes)
        if expand:
            self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1)
            self.bn4 = nn.BatchNorm3d(planes * 4)
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * 4, kernel_size=1, stride=2),
                nn.BatchNorm3d(planes * 4)
            )
        else:
            self.conv4 = nn.Conv3d(planes, inplanes, kernel_size=1)
            self.bn4 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def ST_A(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x

    def ST_B(self, x):
        tmp_x = self.conv2(x)
        tmp_x = self.bn2(tmp_x)
        tmp_x = self.relu(tmp_x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x + tmp_x

    def ST_C(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        tmp_x = self.conv3(x)
        tmp_x = self.bn3(tmp_x)
        tmp_x = self.relu(tmp_x)

        return x + tmp_x

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.ST == 'A':
            out = self.ST_A(out)
        elif self.ST == 'B':
            out = self.ST_B(out)
        elif self.ST == 'C':
            out = self.ST_C(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.expand:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class P3D(nn.Module):

    def __init__(self, block, layers, input_channel=1, config=None):
        super(P3D, self).__init__()
        self.inplanes = config.BACKBONE_CHANNELS[0]

        self.C1 = nn.Sequential(
            nn.Conv3d(input_channel, config.BACKBONE_CHANNELS[0], kernel_size=(3, 7, 7), stride=2, padding=(1, 3, 3)),
            nn.BatchNorm3d(config.BACKBONE_CHANNELS[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )  # size = 1 / 4, channel = config.BACKBONE_CHANNELS[0]

        self.C2 = self._make_layer(block, config.BACKBONE_CHANNELS[0], layers[0], stride=2)  # 1 / 8, [0] * 4
        self.C3 = self._make_layer(block, config.BACKBONE_CHANNELS[1], layers[1], stride=2)  # 1 / 16, [1] * 4

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, 1, True, stride))
        self.inplanes = planes * block.expansion
        for i in range(2, blocks + 1):
            layers.append(block(self.inplanes, planes, i, False))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)

        return x

    def stages(self):
        return [self.C1, self.C2, self.C3]


def P3D19(**kwargs):
    """Construct a P3D19 model."""
    model = P3D(Bottleneck, [2, 3], **kwargs)
    return model
