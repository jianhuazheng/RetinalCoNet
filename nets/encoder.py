import math
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RetinalDualPath(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Magnocellular
        self.magno = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=2, dilation=2),
            nn.InstanceNorm2d(in_channels//2),
            nn.ReLU()
        )
        # Parvocellular
        self.parvo = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.GroupNorm(4, in_channels//2),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU()
        )

    def forward(self, x):
        m_feat = self.magno(x)
        p_feat = self.parvo(x)
        fused = self.fusion(torch.cat([m_feat, p_feat], dim=1))
        return fused + x

class ONOFFParvocellular(nn.Module):

    def __init__(self, in_channels,reduction_ratio=4):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels // 2
        # ON channel
        self.on_center = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU()
        )
        # OFF channel
        self.off_surround = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2),
            nn.GroupNorm(4, out_channels),
            nn.ReLU()
        )

        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, out_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // reduction_ratio, 2, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights()
    def _init_weights(self):

        nn.init.zeros_(self.on_center[0].weight)
        nn.init.zeros_(self.on_center[0].bias)

        on_kernel = torch.full((3, 3), -0.1)
        on_kernel[1, 1] = 0.8

        with torch.no_grad():
            for i in range(self.on_center[0].weight.shape[0]):
                for j in range(self.on_center[0].weight.shape[1]):
                    self.on_center[0].weight[i, j] = on_kernel


        nn.init.zeros_(self.off_surround[0].weight)
        nn.init.zeros_(self.off_surround[0].bias)

        off_kernel = torch.full((3, 3), 0.1)
        off_kernel[1, 1] = -0.8

        with torch.no_grad():
            for out_channel in range(self.off_surround[0].weight.shape[0]):
                for in_channel in range(self.off_surround[0].weight.shape[1]):
                    self.off_surround[0].weight[out_channel, in_channel] = off_kernel
    def forward(self, x):

        on_feat = self.on_center(x)
        off_feat = self.off_surround(x)

        weights = self.fusion_gate(torch.cat([
            F.adaptive_avg_pool2d(on_feat, 1),
            F.adaptive_avg_pool2d(off_feat, 1)
        ], dim=1))

        return weights[:, 0:1] * on_feat + weights[:, 1:2] * off_feat

class InteractiveRetinalDualPath(nn.Module):

    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.in_channels = in_channels

        self.magno = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=2, dilation=2),
            nn.InstanceNorm2d(in_channels // 2),
            nn.ReLU()
        )

        self.parvo = ONOFFParvocellular(in_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU()
        )

        self.cross_transformer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(4, in_channels),
            nn.ReLU()
        )
    def forward(self, x):

        m_feat = self.magno(x)  # [B, C//2, H, W]
        p_feat = self.parvo(x)  # [B, C//2, H, W]

        cross_feat = torch.cat([m_feat, p_feat], dim=1)
        cross_feat = self.cross_transformer(cross_feat)

        fused = torch.cat([m_feat, p_feat], dim=1)
        fused = self.fusion(fused)

        return fused + cross_feat + x



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class RetinalBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__(inplanes, planes, stride, downsample, groups,
                         base_width, dilation, norm_layer)

        self.retinal_path = InteractiveRetinalDualPath(planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)


        return self.retinal_path(out)

class InteractiveEncoderBlock(nn.Module):
    def __init__(self, block, layers, num_classes=1000):

        self.inplanes = 64
        super(InteractiveEncoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change

        self.layer1 = self._make_layer(RetinalBottleneck, 64, layers[0])
        self.layer2 = self._make_layer(RetinalBottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(RetinalBottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(RetinalBottleneck, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)

        x = self.maxpool(feat1)
        feat2 = self.layer1(x)

        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]

def encoderblock(pretrained=False, **kwargs):
    model = InteractiveEncoderBlock(Bottleneck, [3, 4, 6, 3], **kwargs)

    del model.avgpool
    del model.fc
    return model

