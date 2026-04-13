"""
优化器与学习率调度：SGD 分组 + LambdaLR（线性 warmup + 余弦衰减）。
仅在每个 epoch 内全部 optimizer.step() 之后调用一次 scheduler.step()，无参，避免 PyTorch 调度器警告。
"""
from __future__ import annotations

import math

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler, LambdaLR


def build_optimizer(model: nn.Module, base_lr: float, momentum: float, weight_decay: float) -> optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if lname.endswith("bias") or "bn" in lname or "batchnorm" in lname or p.ndim == 1:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return optim.SGD(param_groups, lr=base_lr, momentum=momentum, weight_decay=weight_decay)


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    eta_min: float,
    base_lr: float,
) -> LRScheduler:
    """
    每个 epoch 结束时调用一次 scheduler.step()（无参）。
    - warmup: 第 0 个 epoch 开始前的 lr 由 main 设为 base_lr/warmup_epochs；第 1 轮起由本 lambda 接管。
    - 余弦段覆盖剩余 epoch，与原先 CosineAnnealingLR(T_max=total-warmup) 语义对齐。
    """
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(int(warmup_epochs), 0)
    eta_ratio = (eta_min / base_lr) if base_lr > 0 else 0.0

    def lr_lambda(last_epoch: int) -> float:
        # step() 内先 last_epoch += 1，再 get_lr；此处 last_epoch 为递增后的非负整数
        if warmup_epochs > 0 and last_epoch < warmup_epochs - 1:
            return float(last_epoch + 2) / float(warmup_epochs)
        if warmup_epochs > 0 and last_epoch == warmup_epochs - 1:
            return 1.0

        T = max(total_epochs - warmup_epochs, 1)
        if warmup_epochs == 0:
            T = max(total_epochs, 1)
            s = last_epoch + 1
        else:
            s = last_epoch - warmup_epochs + 1
        s = max(1, min(s, T))
        return eta_ratio + (1.0 - eta_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(s) / float(T)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
