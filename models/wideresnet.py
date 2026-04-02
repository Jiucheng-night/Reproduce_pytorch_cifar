import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate = 0.0 ):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop = nn.Dropout(p = dropRate) if dropRate > 0 else nn.Identity()
        self.equalInOut = (in_planes == out_planes)

        if self.equalInOut:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding = 0, bias=False)
    def forward(self, x):
        identity = x
        # 预激活
        out = self.relu1(self.bn1(x))
        if not self.equalInOut:
            identity = self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.drop(out)
        out = self.conv2(out)
        return identity + out

class NetworkBlock(nn.Module):
    def __init__(self, num_blocks, in_planes, out_planes, block, stride, dropRate = 0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layers(block, in_planes, out_planes, num_blocks, stride, dropRate)

    def _make_layers(self, block, in_planes, out_planes, num_layers, stride, drop_rate):
        layers = []

        for i in range(num_layers):
            in_ch = in_planes if i == 0 else out_planes
            s = stride if i == 0 else 1

            layers.append(block(in_ch, out_planes, s, drop_rate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor, dropRate = 0.0):
        super(WideResNet, self).__init__()

        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6

        nChannels = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        ]
        block = BasicBlock

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.channels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = torch.flatten(out, 1)
        return self.fc(out)










