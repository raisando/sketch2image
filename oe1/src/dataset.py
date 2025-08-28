from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class PairedPix2PixDataset(Dataset):
    def __init__(self, root, split="train", size=256):
        self.files = sorted(Path(root).joinpath(split).glob("*.jpg"))
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        im = Image.open(path).convert("RGB")
        w, h = im.size
        im = self.transform(im)
        # Tomar imagen concatenada y separar
        C, H, W = im.shape
        sketch = im[:, :, :W // 2] * 2 - 1  # normaliza a [-1,1]
        photo  = im[:, :, W // 2:] * 2 - 1
        return {"sketch": sketch, "photo": photo}

def make_loaders(data_root: str, size=256, batch_size=8, num_workers=4, aug=True, gray_sketch=True):
    train_set = PairedPix2PixDataset(data_root, split="train", size=size, aug=aug, to_gray_sketch=gray_sketch)
    val_set = PairedPix2PixDataset(data_root, split="val", size=size, aug=False, to_gray_sketch=gray_sketch)
    test_set = PairedPix2PixDataset(data_root, split="test", size=size, aug=False, to_gray_sketch=gray_sketch)


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
