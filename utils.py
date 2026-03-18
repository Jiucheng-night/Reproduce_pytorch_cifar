import numpy as np
import torch
import sys
import time
import random

# class ToTensor:
#     def __call__(self, img):
#         if isinstance(img, np.ndarray):
#             img = np.transpose(img, (2, 0, 1))
#             img = img.copy()
#             return torch.tensor(img, dtype=torch.float32) / 255.0
#         else:
#             raise TypeError(f"Expected numpy array, got {type(img)}")
#
#
# class Normalize:
#     def __init__(self, mean, std):
#         self.mean = torch.tensor(mean).view(3, 1, 1)
#         self.std = torch.tensor(std).view(3, 1, 1)
#
#     def __call__(self, img):
#         return (img - self.mean) / self.std
# class Compose:
#     def __init__ (self, transforms):
#         self.transforms = transforms
#     def __call__(self, img):
#         for t in self.transforms:
#             img = t(img)
#         return img
#
# class RandomCrop:
#     def __init__ (self, size, padding = 0):
#         self.size = size
#         self.padding = padding
#     def __call__(self, img):
#         if self.padding > 0:
#             img = np.pad(img,
#                          ((self.padding, self.padding),
#                           (self.padding, self.padding),
#                           (0, 0)),
#                          mode = 'constant')
#         h, w, _ = img.shape
#         top = random.randint(0, h - self.size)
#         left = random.randint(0, w-self.size)
#         return img[top:top+self.size, left:left+self.size]
#
# class RandomHorizontalFlip:
#     def __init__ (self, p=0.5):
#         self.p = p
#     def __call__(self, img):
#         if random.random() < self.p:
#             return np.fliplr(img)
#         return img


TOTAL_BAR_LENGTH = 65
last_time = time.time()

def progress_bar(current, total, msg=None):
    global last_time
    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    bar = '=' * cur_len + '.'*(TOTAL_BAR_LENGTH - cur_len - 1)
    sys.stdout.write(f'\r[{bar}]')
    if msg:
        sys.stdout.write(f'\r{msg}')
    if current == total - 1:
        sys.stdout.write('\n')
    sys.stdout.flush()


