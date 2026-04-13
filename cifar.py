import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        """
        root: cifar-10-batches-py 路径
        train: 是否加载训练集
        transform: 数据预处理
        """
        self.root = root
        self.transform = transform
        self.train = train
        self.data = []
        self.labels = []

        if train:
            filelist = [f"data_batch_{i}" for i in range(1,6)]
        else:
            filelist = ["test_batch"]

        for file_name in filelist:
            file_path = os.path.join(root, file_name)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(
                    f"CIFAR-10 数据文件不存在: {file_path}\n"
                )
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding='bytes')
                self.data.append(entry[b'data'])
                self.labels.extend(entry[b'labels'])
        self.data = np.vstack(self.data)
        self.data = self.data.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img = self.data[idx]
        label = int(self.labels[idx])
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        return img, label

class CIFAR100Dataset(Dataset):
    """
    官方 CIFAR-100 Python 版格式（与 torchvision 解压目录一致）。
    root: 含 train、test、meta 的目录，例如 .../cifar-100-python
    """

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train

        file_name = "train" if train else "test"
        file_path = os.path.join(root, file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"CIFAR-100 数据文件不存在: {file_path}\n"
            )

        with open(file_path, "rb") as f:
            entry = pickle.load(f, encoding="bytes")

        self.data = entry[b"data"].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        self.labels = entry[b"fine_labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = int(self.labels[idx])
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label
