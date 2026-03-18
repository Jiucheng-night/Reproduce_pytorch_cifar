from .cnn import SimpleCNN

def get_model(name):
    if name == 'cnn':
        return SimpleCNN()
    else:
        raise ValueError(f'Unknown model: {name}')