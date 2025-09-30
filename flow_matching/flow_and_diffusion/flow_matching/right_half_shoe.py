from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch, random

class RightHalfImages(Dataset):
    """
    Lee *.jpg que son [izq|der], y devuelve SOLO la derecha como imagen [-1,1].
    """
    def __init__(self, root, split="train", size=32):
        self.files = sorted(Path(root).joinpath(split).glob("*.jpg"))
        self.tx = T.Compose([
            T.Resize((size, size*2)),  # alto=size, ancho=2*size
            T.ToTensor(),              # [0,1]
            T.Normalize([0.5]*3, [0.5]*3),  # [-1,1]
        ])
        self.size = size

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        im = Image.open(self.files[i]).convert("RGB")
        im = self.tx(im)                  # [3, size, 2*size]
        _, H, W = im.shape
        x = im[:, :, W//2:]               # derecha → [3, size, size]
        return {"x": x, "path": str(self.files[i])}
