import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List, Optional, Union


# 基本模块的构建
class BasicBlock(nn.Module):
    # 扩展系数，1代表该块输出通道等于planes x 1
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1) :
        super(BasicBlock, self).__init__()
        # 第一层卷积层 stride=stride 为1时H，W不变，为2时通常减半，下采样; bias=False 后面要归一化
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        # 归一化
        self.bn1 = nn.BatchNorm2d(planes)

        #第二层卷积层
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        # shortcut 路径建立
        self.shortcut = nn.Sequential() # 恒等变换
        # 如果尺寸不匹配或者通道不匹配都需要进行修改， kernel_size=1可以方便的修改shape
        if stride !=1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * self.expansion)
            )
    # 定义前向传播函数
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)   # 这一步再进行激活使得残差学习更稳定，是经典 ResNet 的标准写法
        return out

#  Bottleneck 模块
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes: int, planes: int, stride: int = 1) :
        super(Bottleneck, self).__init__()
        # 第一层卷积层-只改变通道数
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 第一层卷积层-空间特征提取
        # 不改变通道数， stride=stride 可能会下采样，保持HW尺寸
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # 第一层卷积层-改变通道数
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        # shortcut
        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes*self.expansion, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * self.expansion)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x应当是张量，返回值应当是张量
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]], # 传进来的是“块的类型”，比如 BasicBlock 或 Bottleneck
        num_blocks: List[int],  # 四个 stage 各堆多少个 block，比如 ResNet18 的 [2,2,2,2]
        num_classes: int = 10,  # 最后分类输出类别数
        drop_rate: float = 0.0,  # 分类前 dropout 比例
        use_cifar_head: bool = True     # 控制输入头
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.drop_rate = drop_rate

        if use_cifar_head:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 创建4个层组
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 自适应平均池化，把输入的空间维度变成 1x1
        self.dropout = nn.Dropout(p=self.drop_rate) if drop_rate > 0. else nn.Identity() # 随机dropout
        self.fc = nn.Linear(512 * block.expansion, num_classes) # 输出概率
        self._initialize_weights()

    # 将block堆在一起形成层
    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]], # 传入要用的块类型（BasicBlock 或 Bottleneck）
            planes: int, # 这个 stage 的“基础通道数”。
            num_blocks: int, # 这个 stage 里要堆多少个 block
            stride: int # 只用于该 stage 的第一个 block
    )-> nn.Sequential: # 返回 nn.Sequential：说明最后会把 block 列表串联成一个可当作模块调用的东西。
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion # 更新能和后面的block串起来
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    # 初始化网络的可训练参数
    def _initialize_weights(self):
        for m in self.modules(): # 遍历这个模型里所有层/子模块（包括 Conv、BN、Linear、以及这些模块内部的层）
            if isinstance(m, nn.Conv2d): # 判断当前模块是不是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # Kaiming 初始化（He 初始化）; mode='fan_out'：根据输出通道数（出度）来计算方差尺度; nonlinearity='relu'：告诉初始化这是配合 ReLU 用的（推导公式会考虑激活特性）
            elif isinstance(m, nn.Linear): # 如果是全连接层
                nn.init.normal_(m.weight, 0, 0.01) # 用均值 0、标准差 0.01 的正态分布初始化权重
                nn.init.constant_(m.bias, 0) # 把偏置置 0

    # 定义前向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out



def ResNet18(): return ResNet(BasicBlock, [2,2,2,2])
def ResNet34(): return ResNet(BasicBlock, [3,4,6,3])
def ResNet50(): return ResNet(Bottleneck, [3,4,6,3])
def ResNet101(): return ResNet(Bottleneck, [3,4,23,3])
def ResNet152(): return ResNet(Bottleneck, [3,8,36,3])