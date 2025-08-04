import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function

from .encoder import encoderblock


class DynamicPromptBlock(nn.Module):
    def __init__(self, in_channels, prompt_len=5, prompt_size=20):
        super(DynamicPromptBlock, self).__init__()
        self.prompt_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels // 8, prompt_len * in_channels * prompt_size * prompt_size),
            nn.Unflatten(1, (prompt_len, in_channels, prompt_size, prompt_size))
        )
        self.linear_layer = nn.Linear(in_channels, prompt_len)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        B, C, H, W = x.shape
        prompt = self.prompt_generator(x)
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt_weights_expanded = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        prompt = prompt_weights_expanded * prompt
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        weighted_features = self.conv3x3(prompt + x)
        return F.silu(weighted_features * x)

class ConditionalNorm(nn.Module):
    def __init__(self, num_features, cond_channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gamma_net = nn.Conv2d(cond_channels, num_features, 3, padding=1)
        self.beta_net = nn.Conv2d(cond_channels, num_features, 3, padding=1)

    def forward(self, x, edge_cond):
        normalized = self.norm(x)
        gamma = self.gamma_net(edge_cond)  # [B,C,H,W]
        beta = self.beta_net(edge_cond)  # [B,C,H,W]
        return (1 + torch.tanh(gamma)) * normalized + beta

class EdgeConditionGenerator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.edge_conv(x)

class BoundaryEnhance(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.edge_gen = EdgeConditionGenerator(in_channels)
        self.norm = ConditionalNorm(in_channels, in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):

        edge_cond = self.edge_gen(x)  # [B,C,H,W]
        norm_feat = self.norm(x, edge_cond)
        return x + self.conv(norm_feat)


class Up(nn.Module):
    def __init__(self, in_size1, in_size2, out_size,n_directions=8):
        super(Up, self).__init__()
        self.conv1 = nn.Conv2d(in_size1 + in_size2, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.channel_adjust1 = nn.Conv2d(in_size1, in_size1, 1) if in_size1 != in_size1 else nn.Identity()
        self.channel_adjust2 = nn.Conv2d(in_size2, in_size2, 1) if in_size2 != in_size2 else nn.Identity()

        self.boundary_enhance = BoundaryEnhance(out_size)


    def _make_adapter(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity(),
            nn.BatchNorm2d(out_c)
        )
    def forward(self, inputs1, inputs2):
        inputs1 = self.channel_adjust1(inputs1)
        inputs2 = self.channel_adjust2(inputs2)
        inputs2 = self.up(inputs2)
        diffY = inputs1.size()[2] - inputs2.size()[2]
        diffX = inputs1.size()[3] - inputs2.size()[3]
        inputs2 = F.pad(inputs2, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
        outputs = torch.cat([inputs1, inputs2], 1)

        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)

        outputs = self.boundary_enhance(outputs)

        return outputs

class RetinalCoNet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='encoderblock'):
        super(RetinalCoNet, self).__init__()
        self.backbone_name = backbone  # 存储backbone名称

        # 初始化backbone
        self.encoder = encoderblock(pretrained=pretrained)
        in_filters = [64, 256, 512, 1024]
        self.adjust_conv = nn.Conv2d(2048, 1024, 1)  # 调整feat5到1024通道


        out_filters = [64, 128, 256, 512]

        # 实例化 PromptGenBlock
        self.prompt_gen_blocks = nn.ModuleList([
            DynamicPromptBlock(in_channels=in_filters[i], prompt_len=5, prompt_size=20)
            for i in range(len(in_filters))
        ])

        # upsampling
        self.up_concat4 = Up(in_filters[3], 1024, out_filters[3])
        self.up_concat3 = Up(in_filters[2], out_filters[3], out_filters[2])
        self.up_concat2 = Up(in_filters[1], out_filters[2], out_filters[1])
        self.up_concat1 = Up(in_filters[0], out_filters[1], out_filters[0])


        self.up_conv = nn.Sequential(
              nn.UpsamplingBilinear2d(scale_factor=2),
              nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
              nn.ReLU(),
              nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
              nn.ReLU(),
            )


        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        # 使用backbone实例提取特征
        [feat1, feat2, feat3, feat4, feat5] = self.encoder(inputs)
        feat5 = self.adjust_conv(feat5)

        # 在跳跃连接中应用 PromptGenBlock
        feat1 = self.prompt_gen_blocks[0](feat1)
        feat2 = self.prompt_gen_blocks[1](feat2)
        feat3 = self.prompt_gen_blocks[2](feat3)
        feat4 = self.prompt_gen_blocks[3](feat4)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        for param in self.encoder.parameters():
             param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


