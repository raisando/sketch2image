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
import mnitsutils as util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MNISTSampler(nn.Module, util.Sampleable):
    """
    Sampleable wrapper for the MNIST dataset
    """
    def __init__(self):
        super().__init__()
        self.dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")

        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels

# ==== Mmodelo condicional por sketch ====
import torch.nn as nn
from diffusers import UNet2DModel

class CondUNet(nn.Module):
    """
    DDPM condicional por imagen: concatena [x_t, sketch] en el canal.
    - x_t: foto ruidosa [B,3,H,W]py
    - sketch: condición [B,1,H,W] (o [B,3,H,W] si no es gris)
    """
    def __init__(self, sample_size: int = 256, gray_sketch: bool = True):
        super().__init__()
        in_total = 3 + (1 if gray_sketch else 3)
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_total,
            out_channels=3,
            # bloques moderados; ajusta si tu GPU aprieta
            block_out_channels=(128, 256, 256, 512),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, sketch: torch.Tensor) -> torch.Tensor:
        # concatena en canales
        x = torch.cat([x_t, sketch], dim=1)
        return self.unet(x, t).sample  # predicción de ε (ruido)


if __name__ == '__main__' :
    # Change these!
    num_rows = 10
    num_cols = 10
    num_timesteps = 5

    # Initialize our sampler
    sampler = MNISTSampler().to(device)

    # Initialize probability path
    path = util.GaussianConditionalProbabilityPath(
        p_data = MNISTSampler(),
        p_simple_shape = [1, 32, 32],
        alpha = util.LinearAlpha(),
        beta = util.LinearBeta()
    ).to(device)

    # Sample
    num_samples = num_rows * num_cols
    z, _ = path.p_data.sample(num_samples)
    z = z.view(-1, 1, 32, 32)

    # Setup plot
    fig, axes = plt.subplots(1, num_timesteps, figsize=(6 * num_cols * num_timesteps, 6 * num_rows))

    # Sample from conditional probability paths and graph
    ts = torch.linspace(0, 1, num_timesteps).to(device)
    for tidx, t in enumerate(ts):
        tt = t.view(1,1,1,1).expand(num_samples, 1, 1, 1) # (num_samples, 1, 1, 1)
        xt = path.sample_conditional_path(z, tt) # (num_samples, 1, 32, 32)
        grid = make_grid(xt, nrow=num_cols, normalize=True, value_range=(-1,1))
        axes[tidx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        axes[tidx].axis("off")
    plt.show()
