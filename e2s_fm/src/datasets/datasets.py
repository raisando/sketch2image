# src/datasets/coco_text.py
from pathlib import Path
from typing import List, Optional, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


# ----------------------------------------------------------
#          Dataset Sampler
# ----------------------------------------------------------

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


# ----------------------------------------------------------
#          CocoText dataset loader
# ----------------------------------------------------------
class CocoTextDataset(Dataset):
    def __init__(self, images_dir: str, embeds_pt: str, size: int = 128):
        self.images_dir = Path(images_dir)
        self.embeds: Dict[int, torch.Tensor] = torch.load(embeds_pt)  # {image_id: [D]}
        # COCO filename: 12 dígitos (p.ej. 000000123456.jpg)
        self.items: List[Path] = sorted(self.images_dir.glob("*.jpg"))
        # Transform
        self.tf = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
            T.ToTensor(),                    # [0,1]
            T.Lambda(lambda x: x*2.0 - 1.0)  # [-1,1]
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        p = self.items[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)  # [3,H,W] in [-1,1]

        # Image id from filename: e.g., COCO_train2017_000000123456.jpg or 000000123456.jpg
        stem = p.stem
        # robust: keep last 12 digits
        iid = int(stem[-12:])

        y = self.embeds.get(iid, None)
        if y is None:
            # Fallback: vector cero si no hay caption (raro)
            # Mejor: filtrar en preproceso; aquí devolvemos algo válido.
            y = torch.zeros(next(iter(self.embeds.values())).shape)

        return {"x": x, "y": y, "path": str(p), "image_id": iid}



# ----------------------------------------------------------
#          Fashion MNIST dataset loader
# ----------------------------------------------------------

class FashionMNISTLoader(Dataset):
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


# ----------------------------------------------------------
#          CIFAR 10 dataset loader
# ----------------------------------------------------------

class CIFAR10Loader(Dataset):
    """
    Lee imágenes RGB desde una raíz tipo:
      root/
        train/
          0/ 1/ ... 9/
        test/
          0/ ... 9/
    Devuelve {"x": tensor[-1,1] de shape [3,H,W], "path": str, "y": int|None}
    """
    def __init__(self, root: Path, size: int = 32):
        self.root = Path(root)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        self.files: List[Path] = [
            p for p in self.root.rglob("*")
            if p.is_file() and (p.suffix.lower() in exts) and not p.name.startswith(".")
        ]
        self.files.sort()
        self.size = size
        self.resize = T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC)
        self.to_tensor = T.ToTensor()  # [0,1]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")   # 3 canales
        img = self.resize(img)               # (size,size)
        x = self.to_tensor(img)              # [3,H,W] en [0,1]
        x = x * 2.0 - 1.0                    # [-1,1]

        # intentar parsear etiqueta desde la carpeta padre (nombre "0".."9")
        label_str = Path(p).parent.name
        y = None
        try:
            y = int(label_str)
        except Exception:
            y = None

        return {"x": x, "path": str(p), "y": y}


# ----------------------------------------------------------
#          Edges2Shoes (pix2pix) dataset loader
# ----------------------------------------------------------
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
