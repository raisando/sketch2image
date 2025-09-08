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
import mnistutils as util
import model
import unet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize probability path
path = util.GaussianConditionalProbabilityPath(
    p_data = model.MNISTSampler(),
    p_simple_shape = [1, 32, 32],
    alpha = util.LinearAlpha(),
    beta = util.LinearBeta()
).to(device)

# Initialize model
unet = unet.MNISTUNet(
    channels = [32, 64, 128],
    num_residual_layers = 2,
    t_embed_dim = 40,
    y_embed_dim = 40,
)

# Initialize trainer
trainer = util.CFGTrainer(path = path, model = unet, eta=0.1)

# Train!
trainer.train(num_epochs = 5000, device=device, lr=1e-3, batch_size=250)

# saving the model
tosave = 'mnist.pth'
torch.save(unet.state_dict(), tosave)
print('saved!!')

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

