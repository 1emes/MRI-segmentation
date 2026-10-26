"""
Trains the tumor (yes/no) CNN classifier on the preprocessed split.

usage: python train.py --data_dir data/processed --arch resnet18 --output_dir results/resnet18
       python train.py --config schedule_wd_freeze --save_every_epoch    # a named config from configs.py

Keeps the checkpoint with the lowest validation loss (model.pt) and stops early after
--patience epochs without improvement. With --save_every_epoch it also writes
checkpoints/epoch_XXX.pt for every epoch. Writes history.csv, training_curves.png and train_info.json.
"""
from __future__ import annotations
import os
import csv
import json
import math
import time
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from classifier import ARCHS, build_model, get_device, make_loader, predict, set_seed, sigmoid
from configs import CONFIGS, DEFAULTS


def lr_factor(epoch: int, epochs: int, warmup_epochs: int) -> float:
    """Linear warmup for `warmup_epochs`, then cosine decay to zero at `epochs` (0-based epoch)."""
    if epoch < warmup_epochs:
        return (epoch + 1) / (warmup_epochs + 1)
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def set_body_trainable(model: nn.Module, trainable: bool) -> None:
    """Freezes or unfreezes everything but the head; frozen batch norm layers also stop updating their stats."""
    head = set(id(p) for p in model.head_module().parameters())
    for p in model.parameters():
        if id(p) not in head:
            p.requires_grad_(trainable)
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train(trainable)


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, cfg: dict,
        output_dir: str = "", save_every_epoch: bool = False, verbose: bool = True) -> dict:
    """Trains `model` with the hyperparameters in `cfg` (see configs.DEFAULTS).

    Returns {"history", "best_epoch", "best_val_loss", "best_state"}; the model ends with the best weights.
    When `output_dir` is given, saves model.pt (best) and, with `save_every_epoch`, checkpoints/epoch_XXX.pt.
    """
    targets = np.array(train_loader.dataset.targets)
    n_pos, n_neg = int(targets.sum()), int(len(targets) - targets.sum())
    # pos_weight balances the loss between the two classes (more "yes" than "no" images).
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, device=device))

    head_ids = set(id(p) for p in model.head_module().parameters())
    groups = [
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": cfg["lr"]},
        {"params": list(model.head_module().parameters()), "lr": cfg["head_lr"] or cfg["lr"]},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: lr_factor(e, cfg["epochs"], cfg["warmup_epochs"]))
    use_amp = device.type == "cuda"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if save_every_epoch:
            os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    history, best_loss, best_epoch, best_state, bad_epochs = [], float("inf"), 0, None, 0
    for epoch in range(1, cfg["epochs"] + 1):
        frozen = epoch <= cfg["freeze_epochs"]
        model.train()
        set_body_trainable(model, not frozen)
        total_loss, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
            loss = criterion(logits.float(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += ((logits > 0).float() == y).sum().item()
            seen += len(y)
        lrs = [g["lr"] for g in optimizer.param_groups]
        scheduler.step()

        val_y, val_logits = predict(model, val_loader, device)
        val_p = sigmoid(val_logits)
        val_loss = criterion(torch.tensor(val_logits, device=device),
                             torch.tensor(val_y, dtype=torch.float32, device=device)).item()
        row = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "train_acc": correct / seen,
            "val_loss": val_loss,
            "val_acc": float(((val_p >= 0.5) == val_y).mean()),
            "val_auc": float(roc_auc_score(val_y, val_p)),
            "lr": lrs[0] * (not frozen),
            "head_lr": lrs[1],
        }
        history.append(row)

        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if output_dir:
            meta = {"arch": cfg["arch"], "size": cfg["size"], "epoch": epoch, "val": row, "config": cfg}
            if save_every_epoch:
                torch.save({**meta, "state_dict": model.state_dict()},
                           os.path.join(output_dir, "checkpoints", f"epoch_{epoch:03d}.pt"))
            if improved:
                torch.save({**meta, "state_dict": best_state}, os.path.join(output_dir, "model.pt"))
        if verbose:
            print(f"epoch {epoch:3d} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} | "
                  f"val loss {val_loss:.4f} acc {row['val_acc']:.3f} auc {row['val_auc']:.3f}"
                  + (" [frozen]" if frozen else "") + (" *" if improved else ""), flush=True)
        if bad_epochs >= cfg["patience"]:
            if verbose:
                print(f"Early stopping: no val loss improvement for {cfg['patience']} epochs")
            break

    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_loss, "best_state": best_state}


def plot_history(history: list, path: str) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE loss"); ax1.set_title("Loss"); ax1.legend()
    ax2.plot(epochs, [h["train_acc"] for h in history], label="train acc")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="val acc")
    ax2.plot(epochs, [h["val_auc"] for h in history], label="val AUC")
    ax2.set_xlabel("epoch"); ax2.set_ylim(0, 1.02); ax2.set_title("Accuracy / AUC"); ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the brain tumor CNN classifier")
    p.add_argument("--data_dir", "-d", default="data/processed", help="Output of preprocess.py")
    p.add_argument("--config", choices=sorted(CONFIGS), default=None,
                   help="Named config from configs.py; explicit flags below override it")
    p.add_argument("--arch", "-a", choices=ARCHS, default=None, help="Model (default: resnet18)")
    p.add_argument("--output_dir", "-o", default=None, help="Output directory (default: results/<config or arch>)")
    p.add_argument("--epochs", type=int, default=None, help="Maximum number of epochs (default: 60)")
    p.add_argument("--patience", type=int, default=None, help="Early stopping patience (default: 15)")
    p.add_argument("--batch_size", type=int, default=None, help="Batch size (default: 32)")
    p.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-4 resnet18, 1e-3 simplecnn)")
    p.add_argument("--head_lr", type=float, default=None, help="Learning rate of the head (default: --lr)")
    p.add_argument("--weight_decay", type=float, default=None, help="AdamW weight decay (default: 1e-4)")
    p.add_argument("--warmup_epochs", type=int, default=None, help="Linear warmup epochs before the cosine (default: 0)")
    p.add_argument("--freeze_epochs", type=int, default=None, help="Epochs training only the head (default: 0)")
    p.add_argument("--size", type=int, default=None, help="Input size (default: 224)")
    p.add_argument("--no_pretrained", action="store_true", help="Do not load ImageNet weights (resnet18)")
    p.add_argument("--save_every_epoch", action="store_true", help="Also save checkpoints/epoch_XXX.pt every epoch")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (default: 2)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    """DEFAULTS <- named config <- explicit command line flags."""
    cfg = dict(DEFAULTS)
    cfg.update(CONFIGS.get(args.config, {}))
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if cfg["lr"] is None:
        cfg["lr"] = {"resnet18": 1e-4, "simplecnn": 1e-3}[cfg["arch"]]
    cfg["pretrained"] = cfg["pretrained"] and not args.no_pretrained
    return cfg


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    output_dir = args.output_dir or os.path.join("results", args.config or cfg["arch"])
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Config: {json.dumps(cfg)}")

    train_loader = make_loader(args.data_dir, "train", True, cfg["batch_size"], cfg["size"], args.num_workers, args.seed)
    val_loader = make_loader(args.data_dir, "val", False, cfg["batch_size"], cfg["size"], args.num_workers, args.seed)
    targets = np.array(train_loader.dataset.targets)
    print(f"Train: {len(targets)} images ({int((targets == 0).sum())} no, {int(targets.sum())} yes) | "
          f"Val: {len(val_loader.dataset)} images")

    model = build_model(cfg["arch"], pretrained=cfg["pretrained"]).to(device)
    start = time.time()
    out = fit(model, train_loader, val_loader, device, cfg, output_dir, args.save_every_epoch)
    elapsed = time.time() - start
    history, best_epoch = out["history"], out["best_epoch"]

    with open(os.path.join(output_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, os.path.join(output_dir, "training_curves.png"))
    info = {
        "arch": cfg["arch"],
        "config": args.config,
        "best_epoch": best_epoch,
        "best_val_loss": out["best_val_loss"],
        "best_val_acc": history[best_epoch - 1]["val_acc"],
        "best_val_auc": history[best_epoch - 1]["val_auc"],
        "epochs_run": len(history),
        "train_seconds": round(elapsed, 1),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "params": sum(p.numel() for p in model.parameters()),
        "seed": args.seed,
        "hyperparameters": cfg,
        "checkpoints_every_epoch": args.save_every_epoch,
    }
    with open(os.path.join(output_dir, "train_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"Best epoch {best_epoch} (val loss {out['best_val_loss']:.4f}); trained in {elapsed:.0f} s; "
          f"checkpoint: {os.path.join(output_dir, 'model.pt')}")


if __name__ == "__main__":
    main()
