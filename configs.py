"""
Training configurations compared by cv.py. Each one adds a single change to the previous one.

    baseline            the configuration of the first experiment (cosine sized for 60 epochs,
                        so it barely decays before early stopping, weak weight decay, no freezing)
    schedule            cosine sized to the real run: 25 epochs with 2 warmup epochs
    schedule_wd         + AdamW weight decay 5e-2
    schedule_wd_freeze  + head-only training for the first 3 epochs and a 10x larger head learning rate
"""

DEFAULTS = {
    "arch": "resnet18",
    "pretrained": True,
    "epochs": 60,
    "patience": 15,
    "batch_size": 32,
    "lr": None,             # None = 1e-4 for resnet18, 1e-3 for simplecnn
    "head_lr": None,        # None = same as lr
    "weight_decay": 1e-4,
    "warmup_epochs": 0,
    "freeze_epochs": 0,
    "size": 224,
}

CONFIGS = {
    "baseline": {},
    "schedule": {"epochs": 25, "warmup_epochs": 2},
    "schedule_wd": {"epochs": 25, "warmup_epochs": 2, "weight_decay": 5e-2},
    "schedule_wd_freeze": {"epochs": 25, "warmup_epochs": 2, "weight_decay": 5e-2,
                           "freeze_epochs": 3, "head_lr": 1e-3},
}
