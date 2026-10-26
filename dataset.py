"""
Downloads the Kaggle dataset and prints the folder that holds the yes/ and no/ subfolders.

usage: python preprocess.py --data_dir "$(python dataset.py | tail -1)"

The repository already ships a copy of the dataset in classification/, so this step is optional.
"""
import os
import kagglehub

path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")

print("Path to dataset files:", path)
for root, dirs, _ in os.walk(path):
    if {"yes", "no"} <= set(dirs):
        print(root)
        break
