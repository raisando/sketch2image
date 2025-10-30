from pathlib import Path
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
from PIL import Image
from .datasets import CocoTextDataset, ImageDatasetSampler, RightHalfImages, FashionMNISTLoader, CIFAR10Loader, ClassFilteredDataset
from src.tools.utils import random_split_dataset
import numpy as np
import random


def make_loaders(data_root: str, size: int, batch_size: int, to_gray: bool = False, num_workers: int = 8):
    root = Path(data_root)
    train_set = RightHalfImages(root / "train", size=size, to_gray=to_gray)
    val_set   = RightHalfImages(root / "val",   size=size, to_gray=to_gray)
    test_set  = RightHalfImages(root / "test",  size=size, to_gray=to_gray)

    # prints útiles para sanity
    print(f"[ds] train_root={root/'train'} n={len(train_set)}")
    print(f"[ds] val_root={root/'val'} n={len(val_set)}")
    print(f"[ds] test_root={root/'test'} n={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader, test_loader


def make_loaders_mnist(data_root: str, size: int, batch_size: int,
                       num_workers: int = 8,
                       class_filter=None):
    root = Path(data_root)
    train_set = FashionMNISTLoader(root / "train", size=size)
    if class_filter is not None:
        train_set = ClassFilteredDataset(train_set, class_filter)

    print(f"[ds] train_root={root/'train'} n={len(train_set)}")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_loader


def make_loaders_cifar10(
    data_root: str,
    size: int = 32,
    batch_size: int = 64,
    num_workers: int = 8,
    class_filter=None,
    pin_memory: bool = True,
    drop_last: bool = True,
):
    """
    Crea dataloaders para CIFAR-10 exportado a carpetas.
    - data_root: carpeta que contiene 'train/' y 'test/' (cada una con 0..9)
    - size: resize final (CIFAR original es 32)
    - class_filter: lista opcional de labels permitidos (p.ej., [0,1,2] o ["0","1"])
    """
    root = Path(data_root)

    train_set = CIFAR10Loader(root / "train", size=size)
    test_set  = CIFAR10Loader(root / "test",  size=size)

    if class_filter is not None:
        train_set = ClassFilteredDataset(train_set, class_filter)
        test_set  = ClassFilteredDataset(test_set,  class_filter)

    print(f"[ds] train_root={root/'train'} n={len(train_set)} size={size} -> x in [-1,1], RGB")
    print(f"[ds] test_root={root/'test'}   n={len(test_set)}  size={size} -> x in [-1,1], RGB")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader


def make_loaders_coco_text(
    data_root   : str,               # e.g. "data/coco2017/images"
    split       : str = "train",     # "train" o "val"
    size        : int = 128,
    batch_size  : int = 64,
    num_workers : int = 8,
    val_ratio   : float = 0.01,
    embeds_pt   : str = None         # e.g. data/coco2017/cache/clip_caps_train_vitb32.pt
):
    """
    Si split == 'train' y val_ratio>0 -> retorna (train_loader, val_loader)
    En otro caso -> retorna un solo loader (para evaluación o inferencia)
    """
    root = Path(data_root)
    img_dir = root / "images" / (f"{split}2017")
    assert embeds_pt is not None, "embeds_pt requerido"
    ds = CocoTextDataset(str(img_dir), embeds_pt, size=size)

    if split == "train" and val_ratio > 0:
        n = len(ds)
        n_val = min(1, int(n * val_ratio))
        train_ratio = 1.0 - val_ratio
        n_train = n - n_val
        train_set, val_set = random_split_dataset(
            ds, [train_ratio, val_ratio]
        )
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False
        )
        return train_loader, val_loader
    else:
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False
        )
        return loader


def seed_worker(worker_id):
    # Semillas independientes por worker
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_loaders_coco_text_distributed(
    data_root, size, batch_size, num_workers, val_ratio, embeds_pt, distributed=True
):

    embeds_train = embeds_pt + "train2017_5cls.pt"
    embeds_val = embeds_pt + "val2017_5cls.pt"

    train_ds = CocoTextDataset(f"{data_root}/train2017", embeds_pt=embeds_train, size=size)
    val_ds   = CocoTextDataset(f"{data_root}/val2017",   embeds_pt=embeds_val, size=size)

    if distributed and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler   = DistributedSampler(val_ds,   shuffle=True, drop_last=True)
        shuffle_train = False
        shuffle_val   = False
    else:
        train_sampler = None
        val_sampler   = None
        shuffle_train = True
        shuffle_val   = False

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, batch_size // 2),
        shuffle=shuffle_val,
        sampler=val_sampler,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader
