from .cnn import SimpleCNN
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

def get_model(name='cnn'):
    name = name.lower()
    if name == 'cnn':
        from .cnn import SimpleCNN
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
    else:
        raise ValueError(f'Unknown model name {name}')