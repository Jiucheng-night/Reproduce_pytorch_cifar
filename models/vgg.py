import torch
import torch.nn as nn

# 网络结构配置表
cfg = {
    'VGG11':[64, 'M', 128, 'M', 256, 256, 'M', 512, 512,'M', 512, 512, 'M'],
    'VGG13':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512,'M', 512, 512, 'M'],
    'VGG16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10) # 最后的通道数全部都是512

    # 定义前向传播函数
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1) # 展平操作
        out = self.classifier(out)
        return out
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3 # 最开始的图像的RGB三通道
        for x in cfg:
            if x == 'M': # 池化
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else: # 卷积层
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),]
                layers += [nn.BatchNorm2d(x)]
                layers += [nn.ReLU(inplace=True)]
                in_channels = x # 需要改变输入通道来匹配下一个卷积层
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

