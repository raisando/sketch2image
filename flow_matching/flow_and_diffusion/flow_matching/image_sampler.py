import torch, random

class ImageDatasetSampler:
    """
    Interfaz tipo MNISTSampler: .sample(n) -> (x, y)
    Para UNCONDITIONAL devolvemos y=0 siempre (clase dummy).
    x debe estar en [-1,1] y shape (n, C, H, W).
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.C = 3
        self.H = dataset.size
        self.W = dataset.size

    def sample(self, n: int):
        idxs = random.choices(range(len(self.dataset)), k=n)
        xs = [self.dataset[i]["x"] for i in idxs]
        x = torch.stack(xs, dim=0)           # (n,3,H,W) en [-1,1]
        y = torch.zeros(n, dtype=torch.long) # todo-cero: sin condición
        return x, y
