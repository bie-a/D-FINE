"""
faster_block.py - FasterNet 的 FasterBlock 实现
参考: "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks"
核心思想：PConv（Partial Convolution）只对部分通道做卷积，减少冗余计算
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import register       # 导入 register

class PartialConv(nn.Module):
    """
    Partial Convolution (PConv)
    只对部分通道做卷积，其余通道直接恒等映射，减少冗余计算
    """
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 只对前 1/4 通道做卷积（论文默认比例）
        self.n_channels = in_channels // 4
        if self.n_channels == 0:
            self.n_channels = in_channels
        self.conv = nn.Conv2d(
            self.n_channels, self.n_channels,
            kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(self.n_channels)
    
    def forward(self, x):
        c = self.n_channels
        x1, x2 = torch.split(x, [c, x.shape[1] - c], dim=1)
        x1 = self.bn(self.conv(x1))
        # 如果空间尺寸变化了，对 x2 做自适应池化保持尺寸一致
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.adaptive_avg_pool2d(x2, x1.shape[2:])
        return torch.cat([x1, x2], dim=1)


class FasterBlock(nn.Module):
    """
    FasterNet 的 FasterBlock
    结构: 1x1 Conv(升维) + GELU -> PConv(3x3) + GELU -> 1x1 Conv(降维) + Skip
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2):
        super().__init__()
        hidden_channels = int(in_channels * expand_ratio)
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        self.conv2 = PartialConv(hidden_channels, 3, stride, padding=1)
        
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.act = nn.GELU()
        
        self.skip = (stride > 1) or (in_channels != out_channels)
        if self.skip:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x if not self.skip else self.shortcut(x)
        
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.conv2(out))
        out = self.bn3(self.conv3(out))
        
        return self.act(out + identity)

@register()
class FasterNetBackbone(nn.Module):
    def __init__(self, width_mul=1.0, depths=(2, 2, 8, 2), channels=(64, 128, 256, 512)):
        super().__init__()
        channels = [int(c * width_mul) for c in channels]
        
        # 期望的最终输出通道 (对应 encoder 的 in_channels: 256, 512, 1024)
        target_channels = [256, 512, 1024]
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )
        
        self.stages = nn.ModuleList()
        for i, (depth, ch) in enumerate(zip(depths, channels)):
            stride = 2 if i > 0 else 1
            blocks = []
            for j in range(depth):
                s = stride if j == 0 else 1
                prev_ch = channels[i-1] if j == 0 and i > 0 else ch
                blocks.append(FasterBlock(prev_ch, ch, s))
            self.stages.append(nn.Sequential(*blocks))
        
        self.out_indices = (1, 2, 3)  # stage1, stage2, stage3 的输出
        
        # 为每个输出添加 1x1 投影层，将通道数提升到 target_channels
        self.out_projections = nn.ModuleList()
        for i, idx in enumerate(self.out_indices):
            c = channels[idx]   # 原本的输出通道
            self.out_projections.append(
                nn.Conv2d(c, target_channels[i], kernel_size=1, bias=False)
            )
    
    def forward(self, x):
        out = self.stem(x)
        features = []
        proj_idx = 0
        for i, stage in enumerate(self.stages):
            out = stage(out)
            if i in self.out_indices:
                # 先投影到目标通道
                out_proj = self.out_projections[proj_idx](out)
                proj_idx += 1
                features.append(out_proj)
        return features


def fasternet_s(**kwargs):
    """FasterNet-S 配置（与论文中参数量匹配）"""
    return FasterNetBackbone(
        width_mul=1.0,
        depths=(2, 2, 8, 2),
        channels=(64, 128, 256, 512),
        **kwargs
    )