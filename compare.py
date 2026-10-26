"""
Side-by-side test comparison of trained runs: one confusion matrix per run and their ROC curves overlaid.

usage: python compare.py results/final/baseline results/final/schedule_wd -o results/final/test_comparison.png

Reads the predictions.csv and metrics.json that evaluate.py wrote in each run folder.
"""
from __future__ import annotations
import os
import csv
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def load_run(run_dir: str):
    with open(os.path.join(run_dir, "predictions.csv")) as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)
    y = np.array([int(r["label"]) for r in rows])
    p = np.array([float(r["prob_tumor"]) for r in rows])
    pred = np.array([int(r["pred"]) for r in rows])
    return y, p, pred, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare evaluated runs on the test split")
    ap.add_argument("runs", nargs="+", help="Run folders with predictions.csv and metrics.json")
    ap.add_argument("--output", "-o", default="results/final/test_comparison.png", help="Output PNG")
    args = ap.parse_args()

    runs = [(os.path.basename(os.path.normpath(r)), *load_run(r)) for r in args.runs]
    fig, axes = plt.subplots(1, len(runs) + 1, figsize=(4.2 * (len(runs) + 1), 4.2))
    for ax, (name, y, _, pred, m) in zip(axes, runs):
        ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0, 1]), display_labels=["no tumor", "tumor"]).plot(
            ax=ax, cmap="Blues", colorbar=False)
        mm = m["metrics"]
        ax.set_title(f"{name}\nacc {mm['accuracy']:.3f} | F1 {mm['f1']:.3f} | threshold {m['threshold']:.2f}",
                     fontsize=10)
    ax = axes[-1]
    for (name, y, p, _, m), color in zip(runs, COLORS):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC {m['metrics']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#a3a29c", linewidth=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC on the test split", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e4e3df", linewidth=1)
    ax.set_axisbelow(True)
    fig.suptitle(f"Test split ({len(runs[0][1])} images, never used for training or model selection)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output, dpi=120, facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
