"""
Evaluates a trained classifier on a split (test by default).

usage: python evaluate.py --checkpoint results/resnet18/model.pt --data_dir data/processed

Writes metrics.json (accuracy, precision, recall, specificity, F1 and ROC AUC, each with a
95% bootstrap confidence interval), confusion_matrix.png, roc_curve.png and predictions.csv
next to the checkpoint (or to --output_dir).

--threshold val picks the decision threshold on the validation split (the highest one that keeps
recall >= --min_recall there) and only then applies it to --split; --suffix keeps both reports apart:
    python evaluate.py -c results/X/model.pt --threshold val --suffix _valthr
"""
from __future__ import annotations
import os
import csv
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from classifier import build_model, get_device, make_loader, predict, select_threshold, sigmoid


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) == 2 else float("nan"),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, threshold: float, n_boot: int, seed: int) -> dict:
    """95% percentile intervals, resampling the evaluated images with replacement."""
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in compute_metrics(y, p, threshold)}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        for k, v in compute_metrics(y[idx], p[idx], threshold).items():
            samples[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()}


def plot_confusion(y: np.ndarray, pred: np.ndarray, title: str, path: str) -> None:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["no tumor", "tumor"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(y: np.ndarray, p: np.ndarray, auc: float, title: str, path: str) -> None:
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the brain tumor CNN classifier")
    p.add_argument("--checkpoint", "-c", required=True, help="model.pt written by train.py")
    p.add_argument("--data_dir", "-d", default="data/processed", help="Output of preprocess.py")
    p.add_argument("--split", default="test", choices=("train", "val", "test"), help="Split (default: test)")
    p.add_argument("--output_dir", "-o", default=None, help="Output directory (default: checkpoint folder)")
    p.add_argument("--threshold", default="0.5",
                   help="Decision threshold, or 'val' to pick it on the validation split (default: 0.5)")
    p.add_argument("--min_recall", type=float, default=0.95,
                   help="Recall the 'val' threshold must keep on the validation split (default: 0.95)")
    p.add_argument("--suffix", default="", help="Suffix for the output file names (e.g. _valthr)")
    p.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples (default: 2000)")
    p.add_argument("--seed", type=int, default=42, help="Bootstrap seed (default: 42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(ckpt["arch"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    loader = make_loader(args.data_dir, args.split, False, batch_size=64, size=ckpt["size"], num_workers=0)
    y, logits = predict(model, loader, device)
    p = sigmoid(logits)
    if args.threshold == "val":
        val_loader = make_loader(args.data_dir, "val", False, batch_size=64, size=ckpt["size"], num_workers=0)
        val_y, val_logits = predict(model, val_loader, device)
        threshold = select_threshold(val_y, sigmoid(val_logits), args.min_recall)
    else:
        threshold = float(args.threshold)
    pred = (p >= threshold).astype(int)

    metrics = {k: float(v) for k, v in compute_metrics(y, p, threshold).items()}
    ci = bootstrap_ci(y, p, threshold, args.n_boot, args.seed)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    report = {
        "arch": ckpt["arch"],
        "split": args.split,
        "n_images": int(len(y)),
        "n_tumor": int(y.sum()),
        "threshold": threshold,
        "threshold_rule": f"val split, recall >= {args.min_recall}" if args.threshold == "val" else "fixed",
        "best_epoch": ckpt["epoch"],
        "metrics": metrics,
        "ci95": ci,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    with open(os.path.join(output_dir, f"metrics{args.suffix}.json"), "w") as f:
        json.dump(report, f, indent=2)

    title = f"{ckpt['arch']} - {args.split} ({len(y)} images, threshold {threshold:.2f})"
    plot_confusion(y, pred, title, os.path.join(output_dir, f"confusion_matrix{args.suffix}.png"))
    plot_roc(y, p, metrics["roc_auc"], title, os.path.join(output_dir, f"roc_curve{args.suffix}.png"))

    files = [os.path.relpath(path, os.path.join(args.data_dir, args.split)) for path, _ in loader.dataset.samples]
    with open(os.path.join(output_dir, f"predictions{args.suffix}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "prob_tumor", "pred"])
        for row in zip(files, y, p, pred):
            writer.writerow([row[0], int(row[1]), f"{row[2]:.4f}", int(row[3])])

    print(title)
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.3f}  (95% CI {ci[k][0]:.3f}-{ci[k][1]:.3f})")
    print(f"  confusion    TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
