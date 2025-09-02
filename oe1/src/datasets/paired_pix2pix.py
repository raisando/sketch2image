# src/datasets/paired_pix2pix.py
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class PairedPix2PixDataset(Dataset):
    """
    Espera imágenes concatenadas horizontalmente: [sketch | photo]
    p. ej. 256x512 (HxW) donde cada mitad es 256x256.
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        size: int = 256,
        aug: bool = True,
        to_gray_sketch: bool = True,
    ):
        self.files = sorted(Path(root).joinpath(split).glob("*.jpg"))
        self.size = size
        self.to_gray_sketch = to_gray_sketch

        base = [T.Resize((size, size * 2))]

        # Importante: aplicamos augmentations a la imagen completa *antes* de separar
        if aug:
            base += [
                T.RandomHorizontalFlip(p=0.5),
                # agrega más si quieres (color jitter NO recomendado antes de separar si afecta solo una mitad)
            ]

        base += [T.ToTensor()]  # [0,1]

        self.transform = T.Compose(base)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        im = Image.open(path).convert("RGB")
        # 1) resize/flip/etc. en la imagen completa
        im = self.transform(im)  # CxHxW en [0,1]
        C, H, W = im.shape
        assert W % 2 == 0, f"El ancho no es par, no puedo separar mitades: {path}, shape={im.shape}"

        # 2) separar mitades
        sketch = im[:, :, : W // 2]
        photo  = im[:, :, W // 2 :]

        # 3) (opcional) sketch a gris manteniendo 3 canales
        if self.to_gray_sketch:
            # convertir a luminancia y repetir a 3 canales
            # weights de RGB (ITU-R BT.601): 0.2989, 0.5870, 0.1140
            r, g, b = sketch[0:1], sketch[1:2], sketch[2:3]
            # after (true 1-channel for --gray_sketch=True)
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            sketch = gray  # [1, H, W]


        # 4) normalizar a [-1, 1]
        sketch = sketch * 2.0 - 1.0
        photo  = photo  * 2.0 - 1.0

        return {"sketch": sketch, "photo": photo, "path": str(path)}

def make_loaders(
    data_root: str,
    size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    aug: bool = True,
    gray_sketch: bool = True,
):
    train_set = PairedPix2PixDataset(data_root, split="train", size=size, aug=aug, to_gray_sketch=gray_sketch)
    val_set   = PairedPix2PixDataset(data_root, split="val",   size=size, aug=False, to_gray_sketch=gray_sketch)
    test_set  = PairedPix2PixDataset(data_root, split="test",  size=size, aug=False, to_gray_sketch=gray_sketch)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # prueba rápida: cambia la ruta a tu dataset
    dl, _, _ = make_loaders("/home/raisando/tesis/oe1/data/edges2shoes", size=256, batch_size=2)
    batch = next(iter(dl))
    print(batch["sketch"].shape, batch["photo"].shape, batch["path"][:2])
