"""
Preprocessing and train/val/test split for the brain MRI classification dataset.

usage: python preprocess.py --data_dir classification --output_dir data/processed

Steps:
    1. Loads every image in <data_dir>/yes and <data_dir>/no as grayscale.
    2. Drops byte-identical copies (the Kaggle dataset ships several of them).
    3. Groups near-duplicates (same 8x8 difference hash), so a group never spans two splits
       and the test set never contains an image the model has already seen.
    4. Crops each image to the brain (largest external contour), pads it to a square and
       resizes it to --size x --size.
    5. Makes a stratified, group-aware split (default 70/15/15) and writes
       <output_dir>/{train,val,test}/{yes,no}/*.png, splits.csv and summary.json.
"""
from __future__ import annotations
import os
import csv
import json
import shutil
import hashlib
import argparse
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple

CLASSES = ("no", "yes")
SPLITS = ("train", "val", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(data_dir: str) -> List[Tuple[str, str]]:
    """Returns (path, label) pairs for every image in <data_dir>/<class>."""
    items = []
    for label in CLASSES:
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for name in sorted(os.listdir(class_dir)):
            if name.lower().endswith(IMAGE_EXTS):
                items.append((os.path.join(class_dir, name), label))
    return items


def load_gray(path: str) -> np.ndarray:
    """Loads any image (RGB, RGBA, palette, grayscale) as a uint8 grayscale array."""
    return np.asarray(Image.open(path).convert("L"))


def dhash(gray: np.ndarray) -> bytes:
    """8x8 difference hash: equal hashes mean visually (near) identical images."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return np.packbits(small[:, 1:] > small[:, :-1]).tobytes()


def crop_brain(gray: np.ndarray, threshold: int = 45) -> np.ndarray:
    """Crops the image to the bounding box of the largest bright contour (the head).

    Falls back to the full image when no contour is found.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.erode(binary, None, iterations=2)
    binary = cv2.dilate(binary, None, iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w < 0.2 * gray.shape[1] or h < 0.2 * gray.shape[0]:
        return gray
    return gray[y:y + h, x:x + w]


def pad_and_resize(gray: np.ndarray, size: int) -> np.ndarray:
    """Pads the image with black borders to a square and resizes it, keeping the aspect ratio."""
    h, w = gray.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = gray
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.parent[self.find(i)] = self.find(j)


def group_near_duplicates(hashes: List[bytes]) -> List[int]:
    """Returns a group id per image; images sharing a hash end up in the same group."""
    uf = UnionFind(len(hashes))
    first_seen: Dict[bytes, int] = {}
    for i, h in enumerate(hashes):
        if h in first_seen:
            uf.union(i, first_seen[h])
        else:
            first_seen[h] = i
    return [uf.find(i) for i in range(len(hashes))]


def split_groups(groups: Dict[int, List[int]], val_frac: float, test_frac: float,
                 rng: np.random.Generator) -> Dict[int, str]:
    """Assigns whole groups (of a single class) to test, val and train until each quota is met."""
    group_ids = list(groups)
    rng.shuffle(group_ids)
    n_total = sum(len(groups[g]) for g in group_ids)
    quotas = [("test", round(n_total * test_frac)), ("val", round(n_total * val_frac))]
    assignment, filled, q = {}, 0, 0
    for g in group_ids:
        while q < len(quotas) and filled >= quotas[q][1]:
            q, filled = q + 1, 0
        assignment[g] = quotas[q][0] if q < len(quotas) else "train"
        filled += len(groups[g])
    return assignment


def ensure_clean_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess and split the brain MRI dataset")
    p.add_argument("--data_dir", "-d", default="classification",
                   help="Folder with the yes/ and no/ subfolders (default: classification)")
    p.add_argument("--output_dir", "-o", default="data/processed", help="Output directory")
    p.add_argument("--size", type=int, default=224, help="Output image side in pixels (default: 224)")
    p.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction (default: 0.15)")
    p.add_argument("--test_frac", type=float, default=0.15, help="Test fraction (default: 0.15)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for the split (default: 42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    items = list_images(args.data_dir)
    print(f"Found {len(items)} images in {args.data_dir}")

    # 1-2. Drop byte-identical copies, keeping the first file of each set.
    seen_md5: Dict[str, str] = {}
    kept, dropped = [], []
    for path, label in items:
        with open(path, "rb") as f:
            digest = hashlib.md5(f.read()).hexdigest()
        if digest in seen_md5:
            dropped.append((path, seen_md5[digest]))
            continue
        seen_md5[digest] = path
        kept.append((path, label))
    print(f"Dropped {len(dropped)} byte-identical copies")

    # 3. Group near-duplicates.
    grays = [load_gray(path) for path, _ in kept]
    groups = group_near_duplicates([dhash(g) for g in grays])
    n_groups = len(set(groups))
    print(f"{len(kept)} unique files in {n_groups} near-duplicate groups")

    # 5. Stratified, group-aware split.
    rng = np.random.default_rng(args.seed)
    split_of = [""] * len(kept)
    for label in CLASSES:
        by_group: Dict[int, List[int]] = {}
        for i, (_, lab) in enumerate(kept):
            if lab == label:
                by_group.setdefault(groups[i], []).append(i)
        for g, split in split_groups(by_group, args.val_frac, args.test_frac, rng).items():
            for i in by_group[g]:
                split_of[i] = split
    labels_of_group: Dict[int, set] = {}
    for i, (_, label) in enumerate(kept):
        labels_of_group.setdefault(groups[i], set()).add(label)
    for g, labels in labels_of_group.items():
        if len(labels) > 1:
            print(f"WARNING: near-duplicate group {g} mixes classes {sorted(labels)}")

    # 4. Crop, resize and write.
    ensure_clean_dir(args.output_dir)
    rows, used_names = [], set()
    for i, ((path, label), gray) in enumerate(zip(kept, grays)):
        out = pad_and_resize(crop_brain(gray), args.size)
        stem = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
        name, n = f"{stem}.png", 1
        # Different files can share a stem (e.g. Y92.jpg and Y92.png).
        while (label, name) in used_names:
            n += 1
            name = f"{stem}_{n}.png"
        used_names.add((label, name))
        out_dir = os.path.join(args.output_dir, split_of[i], label)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, name), out)
        rows.append({"split": split_of[i], "label": label, "file": f"{split_of[i]}/{label}/{name}",
                     "source": path, "group": groups[i]})

    with open(os.path.join(args.output_dir, "splits.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = {s: {c: sum(r["split"] == s and r["label"] == c for r in rows) for c in CLASSES} for s in SPLITS}
    summary = {
        "source_images": len(items),
        "byte_identical_dropped": len(dropped),
        "unique_images": len(kept),
        "near_duplicate_groups": n_groups,
        "size": args.size,
        "seed": args.seed,
        "counts": counts,
        "dropped": [{"file": a, "copy_of": b} for a, b in dropped],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'split':<6} {'no':>4} {'yes':>4} {'total':>6}")
    for s in SPLITS:
        print(f"{s:<6} {counts[s]['no']:>4} {counts[s]['yes']:>4} {sum(counts[s].values()):>6}")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
