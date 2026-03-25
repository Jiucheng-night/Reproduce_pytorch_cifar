import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义块结构
class Block(nn.Module):
    """Depthwise conv + Pointwise conv"""
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        # 下面实现了深度分离卷积
        # groups=in_planes 每个通道分开卷积，不改变通道数，使用3X3的卷积核
        self.conv1 = nn.Conv2d(
            in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_planes)
        # 采用1X1的卷积核将三个不同的通道混合在一起
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNet(nn.Module):
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0] # 判断x是否是整数类型，如果是就等于x，否则取x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1) # 除了通道数，其余的全部展平
        out = self.linear(out)
        return out
