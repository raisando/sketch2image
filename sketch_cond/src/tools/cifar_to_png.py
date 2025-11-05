#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
from tqdm import tqdm

def unpickle(file: Path):
    with open(file, "rb") as fo:
        # CIFAR-10 python version uses bytes keys in py3; encoding='bytes' is safe.
        d = pickle.load(fo, encoding="bytes")
    return d

def save_split(split_name: str, batch_files: List[Path], out_root: Path):
    """
    split_name: 'train' or 'test'
    batch_files: list of .pkl batch files to read
    out_root: root output, we make out_root/split_name/<0..9>/*.png
    """
    split_dir = out_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # make class folders 0..9
    for c in range(10):
        (split_dir / str(c)).mkdir(parents=True, exist_ok=True)

    idx_global = 0
    for batch_path in batch_files:
        batch = unpickle(batch_path)
        data = batch[b"data"]            # shape: (10000, 3072) uint8
        labels = batch[b"labels"]        # list of length 10000 (ints 0..9)

        # reshape to (N, 3, 32, 32) then -> (N, 32, 32, 3) for Pillow
        imgs = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        for i, (img_arr, label) in enumerate(zip(imgs, labels)):
            img = Image.fromarray(img_arr, mode="RGB")
            # filename: zero-padded incremental index; you can also use batch filenames if you prefer
            out_path = split_dir / str(label) / f"{split_name}_{idx_global:06d}.png"
            img.save(out_path, format="PNG")
            idx_global += 1

def main():
    parser = argparse.ArgumentParser(description="Export CIFAR-10 python batches to PNGs (numeric class folders).")
    parser.add_argument("--src", type=str, default="cifar-10-batches-py",
                        help="Path to extracted CIFAR-10 python directory (contains data_batch_1..5, test_batch).")
    parser.add_argument("--out", type=str, default="images",
                        help="Output directory to create (will contain train/ and test/).")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    # collect train and test batch files
    train_batches = [src / f"data_batch_{i}" for i in range(1, 6)]
    test_batches  = [src / "test_batch"]

    # sanity checks
    for p in train_batches + test_batches:
        if not p.exists():
            raise FileNotFoundError(f"Missing expected batch file: {p}")

    # export
    print(f"Writing train images to: {out / 'train'}")
    save_split("train", train_batches, out)

    print(f"Writing test images to: {out / 'test'}")
    save_split("test", test_batches, out)

    # Optional: show a quick summary
    # (we avoid extra imports; you can tree the folder after)
    print("Done. Example structure:\n"
          f"{out}/\n"
          f"  ├─ train/\n"
          f"  │   ├─ 0/ ...\n"
          f"  │   ├─ 1/ ...\n"
          f"  │   └─ 9/ ...\n"
          f"  └─ test/\n"
          f"      ├─ 0/ ...\n"
          f"      └─ 9/ ...")

if __name__ == "__main__":
    main()
