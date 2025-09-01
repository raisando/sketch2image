# src/utils/device.py
import torch

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon (Metal Performance Shaders)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
