"""
Group-aware, stratified 5-fold cross-validation to compare the training configurations in configs.py.

usage: python cv.py --config schedule_wd --seeds 42 43 44 --output_dir results/cv
       python cv.py --summarize results/cv

Only the train + val images of preprocess.py are used; the test split stays untouched for the final
evaluation. For every seed the pool is split into 5 folds with StratifiedGroupKFold (near-duplicate
groups never cross folds). In each fold, the model is trained on 4 folds (with an inner, group-aware
validation split for early stopping and for picking the decision threshold) and evaluated on the 5th,
with the fixed 0.5 threshold and with the threshold chosen on the inner validation split.

--summarize aggregates every <output_dir>/<config>/folds.csv into summary.json, summary.md,
cv_comparison.png and best_config.json (highest mean ROC AUC; the rule is fixed before the runs).
"""
from __future__ import annotations
import os
import csv
import json
import time
import argparse
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from classifier import build_model, get_device, make_file_loader, predict, select_threshold, set_seed, sigmoid
from configs import CONFIGS, DEFAULTS
from evaluate import compute_metrics
from preprocess import split_groups
from train import fit

METRICS = ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc")


def load_pool(data_dir: str):
    """Returns files, labels and groups of the train + val images listed in splits.csv."""
    with open(os.path.join(data_dir, "splits.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in ("train", "val")]
    files = np.array([r["file"] for r in rows])
    labels = np.array([int(r["label"] == "yes") for r in rows])
    groups = np.array([int(r["group"]) for r in rows])
    return files, labels, groups


def inner_split(idx: np.ndarray, labels: np.ndarray, groups: np.ndarray, val_frac: float, seed: int):
    """Splits the training indices of a fold into train/val by whole groups, per class."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for label in (0, 1):
        by_group = {}
        for i in idx[labels[idx] == label]:
            by_group.setdefault(int(groups[i]), []).append(int(i))
        for g, split in split_groups(by_group, val_frac, 0.0, rng).items():
            (val_idx if split == "val" else train_idx).extend(by_group[g])
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


def run_config(args: argparse.Namespace) -> None:
    cfg = dict(DEFAULTS)
    cfg.update(CONFIGS[args.config])
    if args.epochs:
        cfg["epochs"] = args.epochs
    cfg["lr"] = cfg["lr"] or {"resnet18": 1e-4, "simplecnn": 1e-3}[cfg["arch"]]
    out_dir = os.path.join(args.output_dir, args.config)
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    files, labels, groups = load_pool(args.data_dir)
    print(f"[{args.config}] {len(files)} images ({int((labels == 0).sum())} no, {int(labels.sum())} yes), "
          f"{len(set(groups))} groups, config {json.dumps(cfg)}", flush=True)

    rows, oof = [], []
    for seed in args.seeds:
        folds = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (fit_idx, test_idx) in enumerate(folds.split(files, labels, groups)):
            tr_idx, val_idx = inner_split(fit_idx, labels, groups, args.val_frac, seed * 100 + fold)
            set_seed(seed)
            loader = lambda idx, train: make_file_loader(args.data_dir, files[idx].tolist(), labels[idx].tolist(),
                                                         train, cfg["batch_size"], cfg["size"], args.num_workers, seed)
            train_loader, val_loader, test_loader = loader(tr_idx, True), loader(val_idx, False), loader(test_idx, False)
            model = build_model(cfg["arch"], pretrained=cfg["pretrained"]).to(device)
            start = time.time()
            out = fit(model, train_loader, val_loader, device, cfg, verbose=False)

            val_y, val_logits = predict(model, val_loader, device)
            threshold = select_threshold(val_y, sigmoid(val_logits), args.min_recall)
            test_y, test_logits = predict(model, test_loader, device)
            test_p = sigmoid(test_logits)
            row = {"config": args.config, "seed": seed, "fold": fold, "n_train": len(tr_idx), "n_val": len(val_idx),
                   "n_test": len(test_idx), "best_epoch": out["best_epoch"], "epochs_run": len(out["history"]),
                   "seconds": round(time.time() - start, 1), "val_threshold": round(threshold, 4)}
            for tag, thr in (("t05", 0.5), ("tval", threshold)):
                for k, v in compute_metrics(test_y, test_p, thr).items():
                    row[f"{k}_{tag}"] = round(float(v), 4)
            rows.append(row)
            # Out-of-fold predictions: every pool image is predicted once per seed, by the model that never saw it.
            oof += [{"config": args.config, "seed": seed, "fold": fold, "file": f, "label": int(y),
                     "prob_tumor": round(float(p), 5), "val_threshold": round(threshold, 4)}
                    for f, y, p in zip(files[test_idx], test_y, test_p)]
            print(f"[{args.config}] seed {seed} fold {fold}: auc {row['roc_auc_t05']:.3f} | "
                  f"t=0.5 acc {row['accuracy_t05']:.3f} f1 {row['f1_t05']:.3f} | "
                  f"t={threshold:.2f} acc {row['accuracy_tval']:.3f} f1 {row['f1_tval']:.3f} | "
                  f"best epoch {out['best_epoch']}/{len(out['history'])} in {row['seconds']:.0f} s", flush=True)
            del model
            torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "folds.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, "oof.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(oof[0]))
        writer.writeheader()
        writer.writerows(oof)
    print(f"[{args.config}] saved {len(rows)} runs to {out_dir}/folds.csv and {len(oof)} predictions to oof.csv",
          flush=True)


def summarize(cv_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {}
    for name in CONFIGS:  # keeps the order of configs.py
        path = os.path.join(cv_dir, name, "folds.csv")
        if os.path.exists(path):
            with open(path) as f:
                runs[name] = [{k: (float(v) if k != "config" else v) for k, v in r.items()} for r in csv.DictReader(f)]
    if not runs:
        raise SystemExit(f"No folds.csv found under {cv_dir}")

    base = runs.get("baseline")
    summary = {}
    for name, rows in runs.items():
        entry = {"runs": len(rows), "mean_best_epoch": float(np.mean([r["best_epoch"] for r in rows])),
                 "mean_val_threshold": float(np.mean([r["val_threshold"] for r in rows]))}
        for tag in ("t05", "tval"):
            for m in METRICS:
                vals = np.array([r[f"{m}_{tag}"] for r in rows])
                entry[f"{m}_{tag}"] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
        if base is not None and name != "baseline":
            # Same seeds and folds for every config, so differences are paired run by run.
            key = lambda r: (r["seed"], r["fold"])
            b = {key(r): r["roc_auc_t05"] for r in base}
            diffs = np.array([r["roc_auc_t05"] - b[key(r)] for r in rows if key(r) in b])
            entry["auc_vs_baseline"] = {"mean_diff": float(diffs.mean()), "std_diff": float(diffs.std(ddof=1)),
                                        "wins": int((diffs > 0).sum()), "ties": int((diffs == 0).sum()),
                                        "losses": int((diffs < 0).sum())}
        summary[name] = entry

    best = max(summary, key=lambda n: summary[n]["roc_auc_t05"]["mean"])
    with open(os.path.join(cv_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(cv_dir, "best_config.json"), "w") as f:
        json.dump({"config": best, "rule": "highest mean ROC AUC over all folds and seeds",
                   "mean_roc_auc": summary[best]["roc_auc_t05"]["mean"]}, f, indent=2)

    fmt = lambda e: f"{e['mean']:.3f} ± {e['std']:.3f}"
    lines = ["| Config | ROC AUC | Accuracy (0.5) | F1 (0.5) | Recall (0.5) | Specificity (0.5) | "
             "Accuracy (val thr.) | F1 (val thr.) | Recall (val thr.) | Specificity (val thr.) | ΔAUC vs baseline (wins/ties/losses) |",
             "|---" * 11 + "|"]
    for name, e in summary.items():
        d = e.get("auc_vs_baseline")
        delta = f"{d['mean_diff']:+.3f} ({d['wins']}/{d['ties']}/{d['losses']})" if d else "-"
        lines.append(f"| {'**' + name + '**' if name == best else name} | {fmt(e['roc_auc_t05'])} | "
                     + " | ".join(fmt(e[f"{m}_{t}"]) for t in ("t05", "tval")
                                  for m in ("accuracy", "f1", "recall", "specificity"))
                     + f" | {delta} |")
    with open(os.path.join(cv_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Best config (highest mean ROC AUC): {best}")

    # Small multiples: one panel per metric, mean ± std per config; threshold 0.5 vs threshold from val.
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    series = (("t05", "threshold 0.5", "#2a78d6", "o"), ("tval", "threshold from val (recall ≥ 0.95)", "#eb6834", "D"))
    panels = (("roc_auc", "ROC AUC"), ("f1", "F1"), ("recall", "Recall (sensitivity)"), ("specificity", "Specificity"))
    names = list(summary)
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.8), sharey=True)
    for ax, (m, title) in zip(axes, panels):
        tags = series[:1] if m == "roc_auc" else series
        for j, (tag, label, color, marker) in enumerate(tags):
            offset = 0 if len(tags) == 1 else (j - 0.5) * 0.28
            means = [summary[n][f"{m}_{tag}"]["mean"] for n in names]
            stds = [summary[n][f"{m}_{tag}"]["std"] for n in names]
            ax.errorbar(np.arange(len(names)) + offset, means, yerr=stds, fmt=marker, color=color, ms=7,
                        elinewidth=2, capsize=0, label=label)
        ax.set_title(title, color=ink, fontsize=11)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8, color=muted)
        ax.set_ylim(0.3, 1.05)
        ax.grid(axis="y", color=grid, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted)
    handles, labels_ = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.text(0.5, 0.015, f"5-fold group CV, {summary[names[0]]['runs']} runs per config (mean ± std)",
             ha="center", fontsize=9, color=muted)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(os.path.join(cv_dir, "cv_comparison.png"), dpi=130, facecolor="white")
    plt.close(fig)
    plot_oof(cv_dir, names)


def plot_oof(cv_dir: str, names: list) -> None:
    """Pooled out-of-fold confusion matrices (both thresholds) and ROC curves, one per config."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve

    oof = {}
    for name in names:
        path = os.path.join(cv_dir, name, "oof.csv")
        if os.path.exists(path):
            with open(path) as f:
                rows = list(csv.DictReader(f))
            oof[name] = (np.array([int(r["label"]) for r in rows]), np.array([float(r["prob_tumor"]) for r in rows]),
                         np.array([float(r["val_threshold"]) for r in rows]), len({r["seed"] for r in rows}))
    if not oof:
        print("No oof.csv found: skipping the pooled confusion matrices and ROC curves")
        return

    n_images, n_seeds = len(next(iter(oof.values()))[0]) // next(iter(oof.values()))[3], next(iter(oof.values()))[3]
    fig, axes = plt.subplots(2, len(oof), figsize=(3.3 * len(oof), 7.6), squeeze=False,
                             gridspec_kw={"hspace": 0.55})
    for j, (name, (y, p, thr, _)) in enumerate(oof.items()):
        for i, (pred, label) in enumerate((((p >= 0.5).astype(int), "threshold 0.5"),
                                           ((p >= thr).astype(int), "threshold from val"))):
            ax = axes[i, j]
            ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0, 1]), display_labels=["no tumor", "tumor"]).plot(
                ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"{name}\n{label}", fontsize=10)
            if j:  # same classes in every column: keep the y labels on the first one only
                ax.set_ylabel("")
                ax.set_yticklabels([])
    fig.suptitle(f"Pooled out-of-fold predictions: {n_images} images x {n_seeds} seeds", fontsize=11)
    fig.savefig(os.path.join(cv_dir, "cv_confusion_matrices.png"), dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for (name, (y, p, _, _)), color in zip(oof.items(), colors):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#a3a29c", linewidth=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"Pooled out-of-fold ROC ({n_images} images x {n_seeds} seeds)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e4e3df", linewidth=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(cv_dir, "cv_roc.png"), dpi=130, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Group-aware k-fold cross-validation of the training configs")
    p.add_argument("--config", choices=sorted(CONFIGS), help="Config from configs.py to cross-validate")
    p.add_argument("--summarize", metavar="CV_DIR", help="Aggregate the folds.csv files under CV_DIR")
    p.add_argument("--data_dir", "-d", default="data/processed", help="Output of preprocess.py")
    p.add_argument("--output_dir", "-o", default="results/cv", help="Output directory (default: results/cv)")
    p.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Seeds (default: 42 43 44)")
    p.add_argument("--val_frac", type=float, default=0.15, help="Inner validation fraction (default: 0.15)")
    p.add_argument("--min_recall", type=float, default=0.95,
                   help="Recall the validation-chosen threshold must keep (default: 0.95)")
    p.add_argument("--epochs", type=int, default=None, help="Override the config's epochs (quick tests only)")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (default: 2)")
    args = p.parse_args()
    if not (args.config or args.summarize):
        p.error("pass --config NAME or --summarize CV_DIR")
    return args


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize(args.summarize)
    else:
        run_config(args)


if __name__ == "__main__":
    main()
