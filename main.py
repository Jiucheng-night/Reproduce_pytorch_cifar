"""
CIFAR 训练入口：解析配置、组装数据与模型、调度训练循环。
默认每轮在官方 test 上评估并按 test Top1 保存 best；若设置 --val-ratio 则改为 val 选模、test 仅最后评一次。
运行: python main.py [参数]
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from config import parse_train_config, set_seed
from data import (
    build_dataloaders,
    build_datasets,
    build_test_dataset,
    build_train_val_test_loaders,
    build_transforms,
    split_train_val_subsets,
)
from engine import ModelEMA, build_model, evaluate, train_one_epoch
from optim import build_lr_scheduler, build_optimizer


def main() -> None:
    cfg = parse_train_config()
    set_seed(cfg.seed)
    cudnn.benchmark = not cfg.deterministic
    torch.backends.cudnn.deterministic = cfg.deterministic

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_epochs = cfg.resolved_epochs()

    use_val_selection = cfg.val_ratio > 0.0
    if use_val_selection and not (0.0 < cfg.val_ratio < 1.0):
        raise ValueError("--val-ratio 必须在 (0, 1) 内；若不要验证集选模请设为 0")

    run_id = f"{cfg.model}_{cfg.dataset}_lr{cfg.lr}_bs{cfg.batch_size}_seed{cfg.seed}"
    run_dir = os.path.join(cfg.log_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    model_name = cfg.model
    checkpoint_path = os.path.join(run_dir, f"{model_name}_best.pth")
    latest_path = os.path.join(run_dir, f"{model_name}_latest.pth")
    log_path = os.path.join(run_dir, f"{model_name}_log.json")
    config_path = os.path.join(run_dir, f"{model_name}_config.json")

    writer: Optional[SummaryWriter] = None
    if cfg.tensorboard:
        writer = SummaryWriter(log_dir=run_dir)

    transform_train, transform_test, _ = build_transforms(cfg.dataset)

    if use_val_selection:
        train_set, val_set, num_classes = split_train_val_subsets(
            cfg.dataset,
            cfg.data_root,
            transform_train,
            transform_test,
            cfg.val_ratio,
            cfg.seed,
        )
        test_set, nc_test = build_test_dataset(cfg.dataset, cfg.data_root, transform_test)
        assert nc_test == num_classes
        trainloader, valloader, testloader = build_train_val_test_loaders(
            train_set,
            val_set,
            test_set,
            batch_size=cfg.batch_size,
            test_batch_size=cfg.test_batch_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
        )
        print(
            f"验证集选模: val_ratio={cfg.val_ratio} | train={len(train_set)} | val={len(val_set)} | "
            f"test={len(test_set)}（训练过程中不评测 test）"
        )
    else:
        trainset, testset, num_classes = build_datasets(
            cfg.dataset, cfg.data_root, transform_train, transform_test
        )
        trainloader, testloader = build_dataloaders(
            trainset,
            testset,
            batch_size=cfg.batch_size,
            test_batch_size=cfg.test_batch_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
        )
        valloader = None
        print("未划分 val（--val-ratio 0）：每轮在 test 上选 best（旧行为）。")

    net = build_model(cfg.model, num_classes, device, cfg.channels_last)

    if cfg.amp and device == "cuda":
        use_amp = True
        scaler: Optional[GradScaler] = GradScaler("cuda")
    else:
        use_amp = False
        scaler = None

    ema: Optional[ModelEMA] = None
    if not cfg.no_ema:
        ema = ModelEMA(net, decay=cfg.ema_decay)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(net, cfg.lr, cfg.momentum, cfg.weight_decay)
    scheduler = build_lr_scheduler(
        optimizer, total_epochs, cfg.warmup_epochs, eta_min=cfg.min_lr, base_lr=cfg.lr
    )

    start_epoch = 0
    best_acc = 0.0
    history: List[Dict[str, Any]] = []

    saved_cfg = cfg.to_dict()
    saved_cfg["num_classes"] = num_classes
    saved_cfg["use_val_selection"] = use_val_selection
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(saved_cfg, f, indent=2, ensure_ascii=False)

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        net.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc = float(ckpt.get("best_acc", 0.0))
        history = ckpt.get("history", [])
        if use_amp and scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ema is not None and "ema" in ckpt:
            ema.ema.load_state_dict(ckpt["ema"])
        tag = "best_val_top1" if use_val_selection else "best_test_top1"
        print(f"Resumed from {cfg.resume}, start_epoch={start_epoch}, {tag}={best_acc:.2f}%")

    # LambdaLR 首次 step 前：第 0 个 epoch 需手动对齐 warmup 起点（之后仅在有 optimizer.step 后的 epoch 末 step）
    if start_epoch == 0 and cfg.warmup_epochs > 0 and not cfg.resume:
        w = float(cfg.warmup_epochs)
        for g in optimizer.param_groups:
            g["lr"] = float(cfg.lr) / w

    total_start = time.perf_counter()
    for epoch in range(start_epoch, total_epochs):
        lr_current = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{total_epochs} | lr={lr_current:.6f}")
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            net,
            trainloader,
            criterion,
            optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            grad_clip_norm=cfg.grad_clip_norm,
            ema=ema,
            channels_last=cfg.channels_last,
        )
        epoch_time = time.perf_counter() - epoch_start

        eval_model = ema.ema if ema is not None else net

        if use_val_selection:
            assert valloader is not None
            val_res = evaluate(
                eval_model,
                valloader,
                criterion,
                device=device,
                use_amp=use_amp,
                channels_last=cfg.channels_last,
                num_classes=num_classes,
                split_name="Val",
            )
            sel_loss, sel_top1, sel_top5 = val_res.loss, val_res.top1_acc, val_res.top5_acc
            row: Dict[str, Any] = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": val_res.loss,
                "val_top1_acc": val_res.top1_acc,
                "val_top5_acc": val_res.top5_acc,
                "val_precision_macro": val_res.precision_macro,
                "val_recall_macro": val_res.recall_macro,
                "val_f1_macro": val_res.f1_macro,
                "val_precision_weighted": val_res.precision_weighted,
                "val_recall_weighted": val_res.recall_weighted,
                "val_f1_weighted": val_res.f1_weighted,
                "val_balanced_accuracy": val_res.balanced_accuracy,
                "lr": float(lr_current),
                "epoch_time_sec": float(epoch_time),
            }
        else:
            assert valloader is None
            test_res = evaluate(
                eval_model,
                testloader,
                criterion,
                device=device,
                use_amp=use_amp,
                channels_last=cfg.channels_last,
                num_classes=num_classes,
                split_name="Test",
            )
            sel_loss, sel_top1, sel_top5 = test_res.loss, test_res.top1_acc, test_res.top5_acc
            row = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "test_loss": test_res.loss,
                "top1_acc": test_res.top1_acc,
                "top5_acc": test_res.top5_acc,
                "precision_macro": test_res.precision_macro,
                "recall_macro": test_res.recall_macro,
                "f1_macro": test_res.f1_macro,
                "precision_weighted": test_res.precision_weighted,
                "recall_weighted": test_res.recall_weighted,
                "f1_weighted": test_res.f1_weighted,
                "balanced_accuracy": test_res.balanced_accuracy,
                "lr": float(lr_current),
                "epoch_time_sec": float(epoch_time),
            }

        history.append(row)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # 必须在当轮所有 optimizer.step() 之后更新学习率，再保存 checkpoint（resume 时 last_epoch 与 epoch 一致）
        scheduler.step()

        if sel_top1 > best_acc:
            best_acc = sel_top1
            state = {
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "history": history,
                "selection": "val" if use_val_selection else "test",
            }
            if use_amp and scaler is not None:
                state["scaler"] = scaler.state_dict()
            state["scheduler"] = scheduler.state_dict()
            if ema is not None:
                state["ema"] = ema.ema.state_dict()
            torch.save(state, checkpoint_path)
            sel_tag = "val" if use_val_selection else "test"
            print(f"Saving best checkpoint to {checkpoint_path} ({sel_tag} top1={best_acc:.2f}%)")

        if cfg.save_last:
            state = {
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "history": history,
                "selection": "val" if use_val_selection else "test",
            }
            if use_amp and scaler is not None:
                state["scaler"] = scaler.state_dict()
            state["scheduler"] = scheduler.state_dict()
            if ema is not None:
                state["ema"] = ema.ema.state_dict()
            torch.save(state, latest_path)

        if writer is not None:
            ep = epoch + 1
            writer.add_scalar("Loss/train", float(train_loss), ep)
            if use_val_selection:
                vr = val_res
                writer.add_scalar("Loss/val", vr.loss, ep)
                writer.add_scalar("Acc/ValTop1", vr.top1_acc, ep)
                writer.add_scalar("Acc/ValTop5", vr.top5_acc, ep)
                writer.add_scalar("Metrics/Val_Precision_macro", vr.precision_macro, ep)
                writer.add_scalar("Metrics/Val_Recall_macro", vr.recall_macro, ep)
                writer.add_scalar("Metrics/Val_F1_macro", vr.f1_macro, ep)
                writer.add_scalar("Metrics/Val_Precision_weighted", vr.precision_weighted, ep)
                writer.add_scalar("Metrics/Val_Recall_weighted", vr.recall_weighted, ep)
                writer.add_scalar("Metrics/Val_F1_weighted", vr.f1_weighted, ep)
                writer.add_scalar("Metrics/Val_BalancedAcc", vr.balanced_accuracy, ep)
            else:
                tr = test_res
                writer.add_scalar("Loss/test", tr.loss, ep)
                writer.add_scalar("Acc/Top1", tr.top1_acc, ep)
                writer.add_scalar("Acc/Top5", tr.top5_acc, ep)
                writer.add_scalar("Metrics/Precision_macro", tr.precision_macro, ep)
                writer.add_scalar("Metrics/Recall_macro", tr.recall_macro, ep)
                writer.add_scalar("Metrics/F1_macro", tr.f1_macro, ep)
                writer.add_scalar("Metrics/Precision_weighted", tr.precision_weighted, ep)
                writer.add_scalar("Metrics/Recall_weighted", tr.recall_weighted, ep)
                writer.add_scalar("Metrics/F1_weighted", tr.f1_weighted, ep)
                writer.add_scalar("Metrics/BalancedAcc", tr.balanced_accuracy, ep)
            writer.add_scalar("LR", float(lr_current), ep)
            writer.add_scalar("Time/epoch_sec", float(epoch_time), ep)

    if os.path.isfile(checkpoint_path):
        best_state = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(best_state["net"])
        if ema is not None and "ema" in best_state:
            ema.ema.load_state_dict(best_state["ema"])
            final_eval_model = ema.ema
        else:
            final_eval_model = net
    else:
        final_eval_model = ema.ema if ema is not None else net
        print("警告: 未找到 best 权重文件，使用当前模型权重做最终评测（例如未训练或 0 epoch）。")

    final_res = evaluate(
        final_eval_model,
        testloader,
        criterion,
        device=device,
        use_amp=use_amp,
        channels_last=cfg.channels_last,
        num_classes=num_classes,
        split_name="Test",
    )
    fr = final_res
    print(
        f"\n最终 Test（仅评一次）| Loss: {fr.loss:.4f} | Top1: {fr.top1_acc:.2f}% | Top5: {fr.top5_acc:.2f}%"
    )
    print(
        f"  Macro  — P: {fr.precision_macro:.2f}% | R: {fr.recall_macro:.2f}% | F1: {fr.f1_macro:.2f}% | "
        f"BAcc: {fr.balanced_accuracy:.2f}%"
    )
    print(
        f"  Weighted — P: {fr.precision_weighted:.2f}% | R: {fr.recall_weighted:.2f}% | F1: {fr.f1_weighted:.2f}%"
    )

    if writer is not None:
        fe = total_epochs
        writer.add_scalar("Final/TestLoss", fr.loss, fe)
        writer.add_scalar("Final/TestTop1", fr.top1_acc, fe)
        writer.add_scalar("Final/TestTop5", fr.top5_acc, fe)
        writer.add_scalar("Final/Precision_macro", fr.precision_macro, fe)
        writer.add_scalar("Final/Recall_macro", fr.recall_macro, fe)
        writer.add_scalar("Final/F1_macro", fr.f1_macro, fe)
        writer.add_scalar("Final/Precision_weighted", fr.precision_weighted, fe)
        writer.add_scalar("Final/Recall_weighted", fr.recall_weighted, fe)
        writer.add_scalar("Final/F1_weighted", fr.f1_weighted, fe)
        writer.add_scalar("Final/BalancedAcc", fr.balanced_accuracy, fe)
        writer.close()

    total_time = time.perf_counter() - total_start
    if use_val_selection:
        best_key = "val_top1_acc"
        best_label = "Best Val Top1"
    else:
        best_key = "top1_acc"
        best_label = "Best Test Top1"
    best_epoch = max(history, key=lambda x: x.get(best_key, 0.0)).get("epoch", None) if history else None
    print(
        f"Done. {best_label}: {best_acc:.2f}% | Best Epoch: {best_epoch} | "
        f"Total Time: {total_time/60.0:.2f} min"
    )


if __name__ == "__main__":
    main()
