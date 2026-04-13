# models/__init__.py

import re

from .cnn import SimpleCNN
from .mobilenet import MobileNet
from .resnet import ResNet101, ResNet152, ResNet18, ResNet34, ResNet50
from .vgg import VGG
from .wideresnet import WideResNet


def _parse_wide_resnet(name: str):
    """解析 WideResNet 名称，返回 (depth, widen_factor)；无法识别则返回 None。"""
    aliases = {
        "wideresnet": (28, 10),
        "wrn": (28, 10),
        "wrn28_10": (28, 10),
        "wrn-28-10": (28, 10),
        "wrn40_2": (40, 2),
        "wrn-40-2": (40, 2),
    }
    if name in aliases:
        return aliases[name]
    m = re.fullmatch(r"wrn[_-]?(\d+)[_-](\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def get_model(name: str, num_classes: int = 10):
    """
    返回模型实例。num_classes 对分类头生效（CIFAR-10 用 10，CIFAR-100 用 100）。

    名称示例：
        cnn, resnet18, vgg16, mobilenet
        wideresnet / wrn / wrn28_10 / wrn-28-10  -> WRN-28-10
        wrn40_2 / wrn-40-2                       -> WRN-40-2
        wrn{d}_{w} 或 wrn{d}-{w}                  -> 自定义 depth、widen_factor（需满足 (depth-4)%6==0）
    """
    key = name.lower()
    wrn = _parse_wide_resnet(key)
    if wrn is not None:
        depth, widen = wrn
        return WideResNet(depth, num_classes, widen)

    if key == "cnn":
        return SimpleCNN(num_classes=num_classes)
    if key == "resnet18":
        return ResNet18(num_classes=num_classes)
    if key == "resnet34":
        return ResNet34(num_classes=num_classes)
    if key == "resnet50":
        return ResNet50(num_classes=num_classes)
    if key == "resnet101":
        return ResNet101(num_classes=num_classes)
    if key == "resnet152":
        return ResNet152(num_classes=num_classes)
    if key.startswith("vgg"):
        return VGG(name.upper(), num_classes=num_classes)
    if key == "mobilenet":
        return MobileNet(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {name}")
