"""
训练配置：集中管理超参与命令行解析。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class TrainConfig:
    dataset: str = "cifar10"
    model: str = "resnet18"
    data_root: str = "./data"
    epochs: Optional[int] = None
    batch_size: int = 256
    test_batch_size: int = 256
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0
    warmup_epochs: int = 5
    min_lr: float = 0.0
    amp: bool = True
    channels_last: bool = False
    num_workers: int = 2
    seed: int = 42
    save_last: bool = False
    resume: str = ""
    log_dir: str = "./checkpoint"
    val_ratio: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    no_ema: bool = False
    ema_decay: float = 0.999
    grad_clip_norm: float = 0.0
    tensorboard: bool = True
    deterministic: bool = False

    def resolved_epochs(self) -> int:
        if self.epochs is not None:
            return int(self.epochs)
        return 120 if self.dataset == "cifar10" else 200

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_train_config(argv: Optional[list] = None) -> TrainConfig:
    p = argparse.ArgumentParser(description="CIFAR Training (Research-grade Template)")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Model name for models.get_model, e.g. resnet18, vgg16, mobilenet, cnn, wideresnet / wrn28_10.",
    )
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--test-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=5, help="线性 warmup 轮数；0 表示不做 warmup，全程余弦。")
    p.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="余弦退火最小学习率 eta_min（常用 0 或 1e-6 量级）。",
    )
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false", help="禁用 AMP 混合精度。")
    p.add_argument("--channels-last", action="store_true", default=False)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-last", action="store_true", default=False)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--log-dir", type=str, default="./checkpoint")
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="默认 0：每轮在 test 上选 best。若在 (0,1) 内则从训练集划 val，按 val Top1 存 best，test 仅在最后评一次。",
    )
    p.add_argument("--mixup-alpha", type=float, default=0.0)
    p.add_argument("--cutmix-alpha", type=float, default=0.0)
    p.add_argument("--no-ema", action="store_true", default=False)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--grad-clip-norm", type=float, default=0.0, help="0 to disable.")
    p.add_argument("--tensorboard", dest="tensorboard", action="store_true", default=True)
    p.add_argument(
        "--no-tensorboard",
        dest="tensorboard",
        action="store_false",
        help="禁用 TensorBoard 日志写入。",
    )
    p.add_argument("--deterministic", action="store_true", default=False)
    args = p.parse_args(argv)

    return TrainConfig(
        dataset=args.dataset,
        model=args.model,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        amp=args.amp,
        channels_last=args.channels_last,
        num_workers=args.num_workers,
        seed=args.seed,
        save_last=args.save_last,
        resume=args.resume,
        log_dir=args.log_dir,
        val_ratio=args.val_ratio,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        no_ema=args.no_ema,
        ema_decay=args.ema_decay,
        grad_clip_norm=args.grad_clip_norm,
        tensorboard=args.tensorboard,
        deterministic=args.deterministic,
    )


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
