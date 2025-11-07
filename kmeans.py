"""
Script for image segmentation using K-means.

usage: python kmeans.py --input images/input/image.png --clusters 4 --output_dir images/output


Obs: after test, i saw a better segmentation using 4 in cluster arguments. Chage if necessary
"""
from __future__ import annotations
import os
import argparse
import numpy as np
from PIL import Image
from typing import Tuple
from sklearn.cluster import KMeans

def load_image(path: str, max_size: int = 800) -> np.ndarray:
	"""Loads an image and returns an RGB array (H, W, 3).

	Reduces the image if its largest dimension is greater than max_size to speed up K-means.
	"""
	img = Image.open(path).convert("RGB")
	# resize if necessary
	w, h = img.size
	scale = min(1.0, max_size / max(w, h))
	if scale < 1.0:
		new_size = (int(w * scale), int(h * scale))
		img = img.resize(new_size, Image.LANCZOS)
	return np.asarray(img)

def segment_image(arr: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
	"""Segments the image using KMeans over the color space.

	Args:
		arr: image as uint8 array (H, W, 3)
		n_clusters: number of clusters

	Returns:
		segmented: segmented image (H, W, 3) with centroid colors
		labels: label map (H, W) with integers [0, n_clusters-1]
	"""
	h, w, c = arr.shape
	assert c == 3, "Image must have 3 channels (RGB)"

	X = arr.reshape(-1, 3).astype(float)

	# Fit KMeans
	kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
	labels = kmeans.fit_predict(X)
	centers = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)

	segmented = centers[labels].reshape(h, w, 3)
	labels_img = labels.reshape(h, w)
	return segmented, labels_img

def save_image(arr: np.ndarray, path: str) -> None:
	img = Image.fromarray(arr.astype(np.uint8))
	img.save(path)

def save_cluster_masks(labels: np.ndarray, output_dir: str) -> None:
	"""Saves binary masks (PNG) for each cluster."""
	n_clusters = int(labels.max()) + 1
	h, w = labels.shape
	for k in range(n_clusters):
		mask = (labels == k).astype(np.uint8) * 255
		img = Image.fromarray(mask, mode="L")
		img.save(os.path.join(output_dir, f"mask_k_{k}.png"))

def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="K-means Segmentation (image -> clusters)")
	p.add_argument("--input", "-i", required=True, help="Path to the input image")
	p.add_argument("--clusters", "-k", type=int, default=3, help="Number of clusters (default: 3)")
	p.add_argument("--output_dir", "-o", default="outputs", help="Output directory")
	p.add_argument("--max_size", type=int, default=800, help="Largest dimension for resizing (default: 800)")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	ensure_dir(args.output_dir)

	print(f"Loading image: {args.input}")
	img = load_image(args.input, max_size=args.max_size)

	print(f"Running K-means with k={args.clusters}")
	segmented, labels = segment_image(img, n_clusters=args.clusters)

	base = os.path.splitext(os.path.basename(args.input))[0]
	out_segmented = os.path.join(args.output_dir, f"{base}_segmented_k{args.clusters}.png")
	print(f"Saving segmented image to: {out_segmented}")
	save_image(segmented, out_segmented)

	print("Saving cluster masks")
	save_cluster_masks(labels, args.output_dir)


if __name__ == "__main__":
	main()

