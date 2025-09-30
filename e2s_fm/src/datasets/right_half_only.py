from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

class RightHalfImages(Dataset):
    """
    Dataset para pix2pix (A|B concatenadas horizontalmente).
    Devuelve la mitad derecha (B) como tensor en [-1,1], shape [C,H,W].
    """
    def __init__(self, root: str, size: int = 64, to_gray: bool = False):
        self.root = Path(root)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        # listado recursivo; ignora ocultos
        self.files = [
            p for p in self.root.rglob("*")
            if p.is_file() and (p.suffix.lower() in exts) and (not p.name.startswith("."))
        ]
        self.files.sort()
        self.size = size
        self.to_gray = to_gray

        # redimensiona a (H=size, W=2*size) para que corte exacto la mitad
        self.resize = T.Resize((size, 2 * size), interpolation=T.InterpolationMode.BICUBIC)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")              # aseguramos 3 canales
        img = self.resize(img)                             # (H=size, W=2*size)
        img = T.ToTensor()(img)                            # [3, H, 2*W] en [0,1]

        if self.to_gray:
            img = torch.mean(img, dim=0, keepdim=True)     # [1, H, 2*W]

        # cortar mitad derecha (B)
        _, H, W = img.shape
        right = img[:, :, W // 2 :]                        # [C, H, W/2] == [C, size, size]

        # normalizar a [-1,1]
        right = right * 2.0 - 1.0

        return {"x": right, "path": str(path)}

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


import torch
import random

class ImageDatasetSampler:
    """
    Sampler mínimo para datasets de imágenes que entregan dicts con clave 'x' (tensor en [-1,1]).
    Provee .sample(batch_size) -> [B,C,H,W] y expone .dim para el path.
    """
    def __init__(self, dataset, device=None):
        self.dataset = dataset
        x0 = dataset[0]["x"]
        assert isinstance(x0, torch.Tensor), "El dataset debe retornar un tensor en dataset[i]['x']"
        self._shape = tuple(x0.shape)  # (C,H,W)
        self._device = device

    @property
    def dim(self):
        C,H,W = self._shape
        return C * H * W

    def to(self, device):
        self._device = device
        return self

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        n = len(self.dataset)
        idxs = torch.randint(low=0, high=n, size=(batch_size,))
        batch = torch.stack([self.dataset[int(i)]["x"] for i in idxs])
        if self._device is not None:
            batch = batch.to(self._device, non_blocking=True)
        return batch


class FashionMNIST(Dataset):
    def __init__(self, root: Path, size: int = 28):
        self.root = Path(root)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        self.files: List[Path] = [
            p for p in self.root.rglob("*")
            if p.is_file() and (p.suffix.lower() in exts) and not p.name.startswith(".")
        ]
        self.files.sort()
        self.size = size
        self.resize = T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        img = Image.open(p).convert("L")      # 1 canal
        img = self.resize(img)                # (size,size)
        x = T.ToTensor()(img)                 # [1,H,W] en [0,1]
        x = x * 2.0 - 1.0                     # [-1,1]
        return {"x": x, "path": str(p)}


from pathlib import Path
from torch.utils.data import Dataset

class ClassFilteredDataset(Dataset):
    """
    Envuelve un dataset que retorna {"x": tensor, "path": ".../label/filename.png"} y
    filtra por una lista de labels permitidos (enteros o strings).
    """
    def __init__(self, base: Dataset, allowed_labels):
        self.base = base
        self.allowed = set(map(str, allowed_labels))
        # pre-indexar
        keep = []
        for i in range(len(base)):
            p = Path(base[i]["path"])
            label = p.parent.name  # carpeta de la clase
            if label in self.allowed:
                keep.append(i)
        self.idxs = keep

    def __len__(self): return len(self.idxs)

    def __getitem__(self, idx):
        return self.base[self.idxs[idx]]


def make_loaders_mnist(data_root: str, size: int, batch_size: int,
                       num_workers: int = 8,
                       class_filter=None):
    root = Path(data_root)
    train_set = FashionMNIST(root / "train", size=size)
    if class_filter is not None:
        train_set = ClassFilteredDataset(train_set, class_filter)

    print(f"[ds] train_root={root/'train'} n={len(train_set)}")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_loader
