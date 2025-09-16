from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch, random
import torch.nn as nn


class RightHalfImages(Dataset):
    """
    Lee pares [izq|der] y devuelve SOLO la derecha en [-1,1].
    Asume *.jpg dentro de train/ val/ test/.
    """
    def __init__(self, root, split="train", size=32, to_gray=False):
        self.files = sorted(Path(root).joinpath(split).glob("*.jpg"))
        self.size = size
        self.to_gray = to_gray
        tx = [T.Resize((size, size*2))]
        if to_gray:
            tx += [T.Grayscale(num_output_channels=1)]
        tx += [T.ToTensor()]
        if to_gray:
            tx += [T.Normalize([0.5], [0.5])]
        else:
            tx += [T.Normalize([0.5]*3, [0.5]*3)]
        self.tx = T.Compose(tx)

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        im = Image.open(self.files[i]).convert("RGB")
        x = self.tx(im)     # [C, size, 2*size]
        _, H, W = x.shape
        right = x[:, :, W//2:]        # [C, size, size]
        return {"x": right, "path": str(self.files[i])}

class ImageDatasetSampler(nn.Module):
    """
    p_data para Flow Matching:
      - .dim  (C*H*W)
      - .shape (C,H,W)
      - .sample(n) -> x  (solo x, [n,C,H,W] en [-1,1])
      - .to(device) via register_buffer
    """
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        c = 1 if getattr(dataset, "to_gray", False) else 3
        h = getattr(dataset, "size", None)
        w = getattr(dataset, "size", None)
        assert h and w, "Dataset debe exponer .size (int)"

        self.shape = (c, h, w)
        self.dim   = c * h * w

        self.register_buffer("_device_buf", torch.zeros(1), persistent=False)

    def sample(self, n: int):
        idxs = random.choices(range(len(self.dataset)), k=n)
        xs = [self.dataset[i]["x"] for i in idxs]                # [C,H,W] en [-1,1]
        x = torch.stack(xs, dim=0).to(self._device_buf.device)   # [n,C,H,W]
        return x


# src/datasets/right_half_only.py  (añade al final, o crea src/datasets/__init__.py)
from torch.utils.data import DataLoader
import torch

def make_loaders(
    root: str,
    size: int = 64,
    batch_size: int = 64,
    to_gray: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    train_set = RightHalfImages(root, split="train", size=size, to_gray=to_gray)
    val_set   = RightHalfImages(root, split="val",   size=size, to_gray=to_gray)
    test_set  = RightHalfImages(root, split="test",  size=size, to_gray=to_gray)

    pin = pin_memory and torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin, drop_last=False)
    return train_loader, val_loader, test_loader
