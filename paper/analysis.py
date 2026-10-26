"""
Figures and statistics of the paper, computed from results/ (no training).

usage: python paper/analysis.py

Writes paper/fig_test.pdf (confusion matrices and ROC curves on the test split, Portuguese labels)
and prints the numbers used in the tables:

- test proportions (accuracy, precision, sensitivity, specificity) with 95% Wilson intervals, which stay
  informative when a model makes no error of a kind (the bootstrap interval collapses to [1, 1]);
- paired AUC differences between configurations over the same 15 cross-validation runs, with the
  95% interval of the corrected repeated k-fold t-test (Nadeau and Bengio), which accounts for the
  overlap between the training sets of different folds;
- exact McNemar test on the test images where two models disagree.
"""
from __future__ import annotations
import os
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.dirname(os.path.abspath(__file__))

CV_RUNS = {  # config -> folder with folds.csv
    "baseline": "cv/baseline",
    "schedule": "cv/schedule",
    "schedule_wd": "cv/schedule_wd",
    "schedule_wd_freeze": "cv/schedule_wd_freeze",
    "baseline_scratch": "cv_scratch/baseline_scratch",
}
TEST_RUNS = {  # name -> folder with predictions.csv / metrics.json
    "baseline": "final/baseline",
    "schedule_wd": "final/schedule_wd",
    "baseline_scratch": "final/baseline_scratch",
    "simplecnn": "simplecnn",
}
TITLES = {"baseline": "baseline (ImageNet)", "schedule_wd": "schedule_wd (ImageNet)",
          "baseline_scratch": "baseline_scratch (aleatória)", "simplecnn": "CNN simples"}
comma = FuncFormatter(lambda x, _: f"{x:.1f}".replace(".", ","))

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def read_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def corrected_t(diffs: np.ndarray, n_train: float, n_test: float) -> tuple[float, float, float]:
    """Mean and 95% interval of the Nadeau-Bengio corrected t-test for repeated k-fold differences."""
    k = len(diffs)
    var = diffs.var(ddof=1) * (1 / k + n_test / n_train)
    half = stats.t.ppf(0.975, k - 1) * math.sqrt(var)
    return diffs.mean(), diffs.mean() - half, diffs.mean() + half


def mcnemar_exact(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Discordant pairs (only a right, only b right) and the two-sided exact McNemar p-value."""
    only_a = int(((a == y) & (b != y)).sum())
    only_b = int(((b == y) & (a != y)).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(only_a, only_b), n, 0.5))
    return only_a, only_b, p


def cv_tables() -> dict:
    runs = {c: read_csv(os.path.join(RESULTS, p, "folds.csv")) for c, p in CV_RUNS.items()
            if os.path.exists(os.path.join(RESULTS, p, "folds.csv"))}
    key = lambda r: (r["seed"], r["fold"])
    base = {key(r): r for r in runs["baseline"]}
    print("Cross-validation (15 runs, threshold 0.5): mean ± std; paired ΔAUC vs baseline [corrected 95% CI]")
    out = {}
    for c, rows in runs.items():
        by = {key(r): r for r in rows}
        assert set(by) == set(base), c
        row = {m: (np.mean([float(r[m]) for r in rows]), np.std([float(r[m]) for r in rows], ddof=1))
               for m in ("roc_auc_t05", "accuracy_t05", "f1_t05", "recall_t05", "specificity_t05",
                         "recall_tval", "specificity_tval")}
        line = f"{c:19s} " + " ".join(f"{m[:-4]} {v[0]:.3f}±{v[1]:.3f}" for m, v in row.items())
        if c != "baseline":
            d = np.array([float(by[k]["roc_auc_t05"]) - float(base[k]["roc_auc_t05"]) for k in base])
            n_train = np.mean([int(base[k]["n_train"]) + int(base[k]["n_val"]) for k in base])
            n_test = np.mean([int(base[k]["n_test"]) for k in base])
            mean, lo, hi = corrected_t(d, n_train, n_test)
            wins, ties = int((d > 1e-9).sum()), int((np.abs(d) <= 1e-9).sum())
            line += f" | dAUC {mean:+.3f} [{lo:+.3f}; {hi:+.3f}] w/t/l {wins}/{ties}/{len(d) - wins - ties}"
        thr = [float(r["val_threshold"]) for r in rows]
        line += (f" | thr {min(thr):.2f}-{max(thr):.2f} | best epoch {np.mean([int(r['best_epoch']) for r in rows]):.1f}"
                 f" | s/run {min(float(r['seconds']) for r in rows):.0f}-{max(float(r['seconds']) for r in rows):.0f}")
        print(line)
        out[c] = row
    return out


def test_tables() -> dict:
    print("\nTest split (34 images): proportions with 95% Wilson CI, AUC with the bootstrap CI of evaluate.py")
    preds = {}
    for name, path in TEST_RUNS.items():
        f = os.path.join(RESULTS, path, "predictions.csv")
        if not os.path.exists(f):
            continue
        rows = read_csv(f)
        y = np.array([int(r["label"]) for r in rows])
        p = np.array([float(r["prob_tumor"]) for r in rows])
        yhat = np.array([int(r["pred"]) for r in rows])
        preds[name] = ([r["file"] for r in rows], y, p, yhat)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        props = {"acc": (tn + tp, len(y)), "prec": (tp, tp + fp), "sens": (tp, tp + fn), "spec": (tn, tn + fp)}
        with open(os.path.join(RESULTS, path, "metrics.json")) as fh:
            m = json.load(fh)
        cells = [f"{k} {a / b:.3f} [{wilson(a, b)[0]:.2f}; {wilson(a, b)[1]:.2f}]" for k, (a, b) in props.items()]
        lo, hi = m["ci95"]["roc_auc"]
        print(f"{name:17s} " + " ".join(cells) + f" f1 {m['metrics']['f1']:.3f} auc {roc_auc_score(y, p):.3f} "
              f"[{lo:.2f}; {hi:.2f}] TN/FP/FN/TP {tn}/{fp}/{fn}/{tp}")
    for a, b in [("schedule_wd", "baseline"), ("baseline", "baseline_scratch"), ("baseline", "simplecnn"),
                 ("baseline_scratch", "simplecnn")]:
        if a in preds and b in preds:
            assert preds[a][0] == preds[b][0]
            only_a, only_b, pval = mcnemar_exact(preds[a][1], preds[a][3], preds[b][3])
            print(f"McNemar {a} vs {b}: only {a} right {only_a}, only {b} right {only_b}, exact p = {pval:.3f}")
    return preds


def fig_test(preds: dict) -> None:
    names = [n for n in ("baseline", "schedule_wd", "baseline_scratch") if n in preds]
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(7.1, 2.1),
                             gridspec_kw={"width_ratios": [1] * len(names) + [1.3]})
    for ax, name in zip(axes, names):
        _, y, _, yhat = preds[name]
        cm = confusion_matrix(y, yhat, labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=21)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > 12 else "black")
        ax.set_xticks([0, 1], ["sem tumor", "tumor"])
        ax.set_yticks([0, 1], ["sem tumor", "tumor"], rotation=90, va="center")
        ax.set_xlabel("Predito")
        ax.set_title(TITLES[name], fontsize=7.5)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Real")
    ax = axes[-1]
    for name, color in zip(names, ("#2a6fdb", "#e8743b", "#7a7a7a")):
        _, y, p, _ = preds[name]
        fpr, tpr, _ = roc_curve(y, p)
        auc = f"{roc_auc_score(y, p):.3f}".replace(".", ",")
        ax.plot(fpr, tpr, color=color, lw=1.4, label=f"{name} ({auc})")
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=0.8)
    ax.set_xlabel("1 − especificidade")
    ax.set_ylabel("Sensibilidade")
    ax.set_title("Curvas ROC (AUC)")
    ax.xaxis.set_major_formatter(comma)
    ax.yaxis.set_major_formatter(comma)
    ax.legend(loc="lower right", fontsize=6.3, frameon=False)
    ax.grid(color="0.9", lw=0.6)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig_test.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cv_tables()
    preds = test_tables()
    fig_test(preds)
    print(f"\nFigure saved to {OUT}/fig_test.pdf")
