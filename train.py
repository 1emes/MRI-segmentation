"""
Trains the tumor (yes/no) CNN classifier on the preprocessed split.

usage: python train.py --data_dir data/processed --arch resnet18 --output_dir results/resnet18

Keeps the checkpoint with the lowest validation loss (model.pt) and stops early after
--patience epochs without improvement. Writes history.csv and training_curves.png.
"""
from __future__ import annotations
import os
import csv
import json
import time
import argparse
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from classifier import ARCHS, build_model, get_device, make_loader, predict, set_seed, sigmoid

DEFAULT_LR = {"resnet18": 1e-4, "simplecnn": 1e-3}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the brain tumor CNN classifier")
    p.add_argument("--data_dir", "-d", default="data/processed", help="Output of preprocess.py")
    p.add_argument("--arch", "-a", choices=ARCHS, default="resnet18", help="Model (default: resnet18)")
    p.add_argument("--output_dir", "-o", default=None, help="Output directory (default: results/<arch>)")
    p.add_argument("--epochs", type=int, default=60, help="Maximum number of epochs (default: 60)")
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    p.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-4 resnet18, 1e-3 simplecnn)")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    p.add_argument("--size", type=int, default=224, help="Input size (default: 224)")
    p.add_argument("--no_pretrained", action="store_true", help="Do not load ImageNet weights (resnet18)")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (default: 2)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


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


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir or os.path.join("results", args.arch)
    args.lr = args.lr or DEFAULT_LR[args.arch]
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_loader = make_loader(args.data_dir, "train", True, args.batch_size, args.size, args.num_workers, args.seed)
    val_loader = make_loader(args.data_dir, "val", False, args.batch_size, args.size, args.num_workers, args.seed)
    targets = np.array(train_loader.dataset.targets)
    n_pos, n_neg = int(targets.sum()), int(len(targets) - targets.sum())
    print(f"Train: {len(targets)} images ({n_neg} no, {n_pos} yes) | Val: {len(val_loader.dataset)} images")

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    # pos_weight balances the loss between the two classes (more "yes" than "no" images).
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"

    history, best_loss, best_epoch, bad_epochs = [], float("inf"), 0, 0
    ckpt_path = os.path.join(args.output_dir, "model.pt")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
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
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save({"arch": args.arch, "size": args.size, "epoch": epoch,
                        "state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
        else:
            bad_epochs += 1
        print(f"epoch {epoch:3d} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} | "
              f"val loss {val_loss:.4f} acc {row['val_acc']:.3f} auc {row['val_auc']:.3f}" + (" *" if improved else ""),
              flush=True)
        if bad_epochs >= args.patience:
            print(f"Early stopping: no val loss improvement for {args.patience} epochs")
            break

    elapsed = time.time() - start
    with open(os.path.join(args.output_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, os.path.join(args.output_dir, "training_curves.png"))
    info = {
        "arch": args.arch,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "best_val_acc": history[best_epoch - 1]["val_acc"],
        "best_val_auc": history[best_epoch - 1]["val_auc"],
        "epochs_run": len(history),
        "train_seconds": round(elapsed, 1),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "params": sum(p.numel() for p in model.parameters()),
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("data_dir", "output_dir")},
    }
    with open(os.path.join(args.output_dir, "train_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"Best epoch {best_epoch} (val loss {best_loss:.4f}); trained in {elapsed:.0f} s; checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
