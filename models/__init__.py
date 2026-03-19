# models/__init__.py

from .cnn import SimpleCNN
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .vgg import VGG
from .mobilenet import MobileNet

def get_model(name):
    """
    返回模型实例
    name: str, 模型名称，可选：
        'cnn'       -> SimpleCNN
        'resnet18'  -> ResNet18
        'resnet34'  -> ResNet34
        'resnet50'  -> ResNet50
        'resnet101' -> ResNet101
        'resnet152' -> ResNet152
        'vgg11'     -> VGG11
        'vgg13'     -> VGG13
        'vgg16'     -> VGG16
        'vgg19'     -> VGG19
    """
    name = name.lower()
    if name == 'cnn':
        return SimpleCNN()
    elif name == 'resnet18':
        return ResNet18()
    elif name == 'resnet34':
        return ResNet34()
    elif name == 'resnet50':
        return ResNet50()
    elif name == 'resnet101':
        return ResNet101()
    elif name == 'resnet152':
        return ResNet152()
    elif name.startswith('vgg'):
        # 例如 vgg11, vgg16
        return VGG(name.upper())
    elif name == "mobilenet":
        return MobileNet()
    else:
        raise ValueError(f"Unknown model name: {name}")