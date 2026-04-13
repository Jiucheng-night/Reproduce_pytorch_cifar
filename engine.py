"""
训练 / 验证循环、Mixup-CutMix、EMA、指标。
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from models import get_model
from utils import progress_bar


@dataclass
class EvalResult:
    """验证/测试集指标（除 loss 外均为百分比 0–100）。"""

    loss: float
    top1_acc: float
    top5_acc: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    balanced_accuracy: float

    def to_log_dict(self) -> Dict[str, float]:
        d = asdict(self)
        return {k: float(v) for k, v in d.items()}


def _accumulate_confusion_flat(
    outputs: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    pred = outputs.argmax(dim=1).long().view(-1)
    t = targets.long().view(-1)
    idx = t * num_classes + pred
    return torch.bincount(idx, minlength=num_classes * num_classes)


def _metrics_from_confusion(cm: torch.Tensor) -> Tuple[float, float, float, float, float, float, float]:
    """由混淆矩阵得到 macro/weighted P、R、F1 与 balanced accuracy（返回 0–100 标量）。"""
    cm = cm.float()
    row = cm.sum(dim=1)
    col = cm.sum(dim=0)
    diag = torch.diagonal(cm)
    eps = 1e-12
    prec_c = torch.where(col > 0, diag / (col + eps), torch.zeros_like(diag))
    rec_c = torch.where(row > 0, diag / (row + eps), torch.zeros_like(diag))
    f1_c = torch.where(
        prec_c + rec_c > eps,
        2 * prec_c * rec_c / (prec_c + rec_c + eps),
        torch.zeros_like(diag),
    )
    mask = row > 0
    n_macro = mask.sum().clamp(min=1).float()
    macro_p = (prec_c * mask.float()).sum() / n_macro
    macro_r = (rec_c * mask.float()).sum() / n_macro
    macro_f1 = (f1_c * mask.float()).sum() / n_macro
    total = row.sum().clamp(min=1)
    w_p = (prec_c * row).sum() / total
    w_r = (rec_c * row).sum() / total
    w_f1 = (f1_c * row).sum() / total
    bal = rec_c[mask].mean() if bool(mask.any()) else rec_c.new_tensor(0.0)
    return (
        (macro_p * 100).item(),
        (macro_r * 100).item(),
        (macro_f1 * 100).item(),
        (w_p * 100).item(),
        (w_r * 100).item(),
        (w_f1 * 100).item(),
        (bal * 100).item(),
    )


def accuracy_topk(outputs: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[float]:
    maxk = max(topk)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res: List[float] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.item() * 100.0 / targets.size(0))
    return res


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool,
    channels_last: bool,
    num_classes: int,
    split_name: str = "Test",
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total = 0
    top1_sum = 0.0
    top5_sum = 0.0
    cm_flat = torch.zeros(num_classes * num_classes, device=device, dtype=torch.long)
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        if channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        if use_amp and device == "cuda":
            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total += bs
        top1, top5 = accuracy_topk(outputs, targets, topk=(1, 5))
        top1_sum += top1 * bs / 100.0
        top5_sum += top5 * bs / 100.0
        cm_flat += _accumulate_confusion_flat(outputs, targets, num_classes)
        progress_bar(
            batch_idx,
            len(loader),
            f"{split_name} Loss: {total_loss / total:.4f} | Top1: {top1_sum * 100.0 / total:.2f}% | "
            f"Top5: {top5_sum * 100.0 / total:.2f}%",
        )

    if total == 0:
        z = 0.0
        return EvalResult(z, z, z, z, z, z, z, z, z, z)
    top1_avg = top1_sum * 100.0 / total
    top5_avg = top5_sum * 100.0 / total
    cm = cm_flat.view(num_classes, num_classes).cpu()
    macro_p, macro_r, macro_f1, w_p, w_r, w_f1, bacc = _metrics_from_confusion(cm)
    return EvalResult(
        loss=total_loss / total,
        top1_acc=top1_avg,
        top5_acc=top5_avg,
        precision_macro=macro_p,
        recall_macro=macro_r,
        f1_macro=macro_f1,
        precision_weighted=w_p,
        recall_weighted=w_r,
        f1_weighted=w_f1,
        balanced_accuracy=bacc,
    )


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            ema_v.copy_(ema_v * decay + msd[k] * (1.0 - decay))

    def to(self, device: str) -> "ModelEMA":
        self.ema.to(device)
        return self


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def rand_bbox(size: Tuple[int, int, int, int], lam: float) -> Tuple[int, int, int, int]:
    H, W = size[2], size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return int(bbx1), int(bby1), int(bbx2), int(bby2)


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_mixed = x.clone()
    x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    area = (bbx2 - bbx1) * (bby2 - bby1)
    total_area = x.size(2) * x.size(3)
    lam_adjusted = 1.0 - float(area) / float(total_area)
    return x_mixed, y_a, y_b, lam_adjusted


def train_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    scaler: Optional[GradScaler],
    use_amp: bool,
    mixup_alpha: float,
    cutmix_alpha: float,
    grad_clip_norm: float,
    ema: Optional[ModelEMA],
    channels_last: bool,
) -> float:
    net.train()
    total_loss = 0.0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        if channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        use_mixup = mixup_alpha > 0.0
        use_cutmix = cutmix_alpha > 0.0
        if use_mixup and use_cutmix:
            use_mixup = random.random() < 0.5
            use_cutmix = not use_mixup

        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
        elif use_cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha)
        else:
            targets_a, targets_b, lam = targets, targets, 1.0

        if use_amp:
            with autocast("cuda"):
                outputs = net(inputs)
                loss = lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm and grad_clip_norm > 0:
                clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = net(inputs)
            loss = lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        if ema is not None:
            ema.update(net)

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total += bs
        progress_bar(batch_idx, len(loader), f"Train Loss: {total_loss / total:.4f}")

    if total == 0:
        return 0.0
    return total_loss / total


def build_model(model_name: str, num_classes: int, device: str, channels_last: bool) -> nn.Module:
    net = get_model(model_name, num_classes=num_classes).to(device)
    if channels_last:
        net = net.to(memory_format=torch.channels_last)
    return net
