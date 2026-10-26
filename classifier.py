"""
Shared pieces of the tumor classifier: models, transforms and data loading.

The positive class is "yes" (tumor). ImageFolder sorts classes alphabetically, so no=0 and yes=1.
"""
from __future__ import annotations
import os
import random
from typing import List
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

ARCHS = ("resnet18", "simplecnn")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SimpleCNN(nn.Module):
    """Small CNN trained from scratch, used as a baseline for the pretrained ResNet-18."""

    def __init__(self, channels=(32, 64, 128, 256), dropout: float = 0.3) -> None:
        super().__init__()
        layers, in_ch = [], 3
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)

    def head_module(self) -> nn.Module:
        return self.head


class ResNet18Binary(nn.Module):
    """ResNet-18 (optionally ImageNet-pretrained) with a single-logit head."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.fc.in_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)

    def head_module(self) -> nn.Module:
        return self.backbone.fc


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    if arch == "resnet18":
        return ResNet18Binary(pretrained=pretrained)
    if arch == "simplecnn":
        return SimpleCNN()
    raise ValueError(f"Unknown arch '{arch}', choose one of {ARCHS}")


def get_transforms(train: bool, size: int = 224) -> transforms.Compose:
    """Light augmentation for training; deterministic resize + normalization otherwise."""
    if train:
        steps = [
            transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    else:
        steps = [transforms.Resize((size, size))]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class FileListDataset(Dataset):
    """Images given as paths relative to `root`, with their 0/1 labels (used by the cross-validation)."""

    def __init__(self, root: str, files: List[str], labels: List[int], transform=None) -> None:
        self.samples = [(os.path.join(root, f), int(y)) for f, y in zip(files, labels)]
        self.targets = [y for _, y in self.samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        return (self.transform(img) if self.transform else img), y


def make_loader(data_dir: str, split: str, train: bool, batch_size: int, size: int = 224,
                num_workers: int = 2, seed: int = 42) -> DataLoader:
    """Loads <data_dir>/<split>/{no,yes} as grayscale-in-RGB images (ImageFolder converts to RGB)."""
    dataset = datasets.ImageFolder(os.path.join(data_dir, split), transform=get_transforms(train, size))
    assert dataset.classes == ["no", "yes"], f"Unexpected classes: {dataset.classes}"
    return wrap_loader(dataset, train, batch_size, num_workers, seed)


def make_file_loader(root: str, files: List[str], labels: List[int], train: bool, batch_size: int,
                     size: int = 224, num_workers: int = 2, seed: int = 42) -> DataLoader:
    dataset = FileListDataset(root, files, labels, transform=get_transforms(train, size))
    return wrap_loader(dataset, train, batch_size, num_workers, seed)


def wrap_loader(dataset: Dataset, train: bool, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), generator=generator,
                      worker_init_fn=seed_worker, persistent_workers=num_workers > 0)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    """Returns (labels, logits) as numpy arrays; the tumor probability is sigmoid(logit)."""
    model.eval()
    labels, logits = [], []
    for x, y in loader:
        logits.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(logits)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def select_threshold(y: np.ndarray, p: np.ndarray, min_recall: float = 0.95) -> float:
    """Highest threshold that keeps recall >= min_recall on (y, p), i.e. the most specific one under
    that sensitivity constraint. Meant to be fitted on the validation split, never on the test split."""
    pos = np.sort(p[y == 1])
    allowed_misses = int(np.floor((1.0 - min_recall) * len(pos) + 1e-9))
    return float(pos[allowed_misses])
