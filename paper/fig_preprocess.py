"""Generates paper/fig_preprocess.pdf: raw MRI slices and the result of crop_brain + pad_and_resize.

usage: python paper/fig_preprocess.py
"""
import os, sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocess import load_gray, crop_brain, pad_and_resize
root = os.path.join(ROOT, "classification") + "/"
files = [("yes/Y1.jpg", "Tumor"), ("yes/Y20.jpg", "Tumor"), ("no/no 1.jpg", "Sem tumor"), ("no/N2.JPG", "Sem tumor")]
fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.7))
for j, (f, lab) in enumerate(files):
    g = load_gray(root + f)
    out = pad_and_resize(crop_brain(g), 224)
    ax[0, j].imshow(g, cmap="gray"); ax[0, j].set_title(f"{lab}\n{g.shape[1]}x{g.shape[0]} px", fontsize=8)
    ax[1, j].imshow(out, cmap="gray"); ax[1, j].set_title("recorte + 224x224", fontsize=8)
for a in ax.ravel(): a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_preprocess.pdf"), bbox_inches="tight")
