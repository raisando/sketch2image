# flow_matching/flow_and_diffusion/mnist/edges_utils.py
from pathlib import Path
from typing import Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class PairedPix2PixDataset(Dataset):
    """
    Lee pares concatenados [sketch | photo] en un solo JPG/PNG.
    - resize a (H, 2H)
    - separa mitades
    - normaliza a [-1, 1]
    """
    def __init__(self, root: str, split: str = "train", size: int = 256, aug: bool = True, to_gray_sketch: bool = True):
        root = Path(root) / split
        self.files = sorted([*root.glob("*.jpg"), *root.glob("*.png")])
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images in {root} (jpg/png)")
        tfms = [T.Resize((size, size*2), interpolation=T.InterpolationMode.BILINEAR)]
        if aug:
            tfms += [T.RandomHorizontalFlip(0.5)]
        tfms += [T.ToTensor()]  # [0,1]
        self.t = T.Compose(tfms)
        self.gray = to_gray_sketch

    def __len__(self): return len(self.files)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        p = self.files[i]
        im = Image.open(p).convert("RGB")
        im = self.t(im)                      # [3, H, 2H]
        C, H, W = im.shape
        assert W == 2*H, f"Esperaba ancho=2H, got {im.shape} en {p}"

        sketch = im[:, :, :W//2]             # [3, H, H]
        photo  = im[:, :, W//2:]             # [3, H, H]

        if self.gray:
            r, g, b = sketch[0:1], sketch[1:2], sketch[2:3]
            sketch = (0.2989*r + 0.5870*g + 0.1140*b)  # [1, H, H]

        # a [-1,1]
        sketch = sketch*2 - 1
        photo  = photo*2 - 1
        return {"sketch": sketch, "photo": photo, "path": str(p)}


def make_loaders(data_root: str, size=256, batch=4, num_workers=4, gray_sketch=True):
    train = PairedPix2PixDataset(data_root, "train", size, aug=True,  to_gray_sketch=gray_sketch)
    val   = PairedPix2PixDataset(data_root, "val",   size, aug=False, to_gray_sketch=gray_sketch)
    test  = PairedPix2PixDataset(data_root, "test",  size, aug=False, to_gray_sketch=gray_sketch)
    pin = torch.cuda.is_available()
    return (
        DataLoader(train, batch, True,  num_workers=num_workers, pin_memory=pin),
        DataLoader(val,   batch, False, num_workers=num_workers, pin_memory=pin),
        DataLoader(test,  batch, False, num_workers=num_workers, pin_memory=pin),
    )


dl, _, _ = make_loaders("oe1/data/edges2shoes", size=128, batch=2, gray_sketch=True)

b = next(iter(dl))
print(b["sketch"].shape, b["photo"].shape)
