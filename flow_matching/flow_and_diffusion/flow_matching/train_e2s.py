from pathlib import Path
import sys
# añade la carpeta "flow_and_diffusion" al sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
import torch
import torch.nn as nn
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
import seaborn as sns
from sklearn.datasets import make_moons, make_circles
from torchvision import datasets, transforms
from torchvision.utils import make_grid
# en vez de MNIST
from right_half_shoe import RightHalfImages
from image_sampler import ImageDatasetSampler

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon (Metal Performance Shaders)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
print("Using device:", device)

DATA_ROOT = "/Users/raimundosandoval/code/U/sketch2image/sketch2image/oe1/data/edges2shoes"
SIZE = 32       # empieza 32x32; luego puedes subir a 64
EPOCHS = 50     # o el número que prefieras
BATCH = 32     # ajusta según GPU (M2: 32-64; CUDA: 128-256)
LR = 1e-3
OUT = "runs/fm_e2s_uncond_32"

import mnistutils as util    # como en el repo MNIST
import unet as unet_module   # el UNet del repo

trainset = RightHalfImages(DATA_ROOT, split="train", size=SIZE)
sampler = ImageDatasetSampler(trainset)

path = util.GaussianConditionalProbabilityPath(
    p_data = sampler,
    p_simple_shape = [3, SIZE, SIZE],  # usa [1, SIZE, SIZE] si dataset en gris
    alpha = util.LinearAlpha(),
    beta  = util.LinearBeta()
).to(device)

# Si tu UNet acepta in_channels:
unet = unet_module.MNISTUNet(
    channels=[32,64,128],
    num_residual_layers=2,
    t_embed_dim=40,
    y_embed_dim=40,
    in_channels=3,   # 3 si RGB; 1 si gris
).to(device)

trainer = util.CFGTrainer(path=path, model=unet, eta=0.1, out_dir=OUT)  # si tu trainer soporta out_dir

trainer.train(num_epochs=EPOCHS, device=device, lr=LR, batch_size=BATCH)

torch.save(unet.state_dict(), f"{OUT}/edges2shoes_uncond.pth")
print("saved!!")


# # Play with these!
# samples_per_class = 10
# num_timesteps = 100
# guidance_scales = [1.0, 3.0, 5.0]

# # Graph
# fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))

# for idx, w in enumerate(guidance_scales):
#     # Setup ode and simulator
#     ode = util.CFGVectorFieldODE(unet, guidance_scale=w)
#     simulator = util.EulerSimulator(ode)

#     # Sample initial conditions
#     y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
#     num_samples = y.shape[0]
#     x0, _ = path.p_simple.sample(num_samples) # (num_samples, 1, 32, 32)

#     # Simulate
#     ts = torch.linspace(0,1,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
#     x1 = simulator.simulate(x0, ts, y=y)

#     # Plot
#     grid = make_grid(x1, nrow=samples_per_class, normalize=True, value_range=(-1,1))
#     axes[idx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
#     axes[idx].axis("off")
#     axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)
# plt.show()
