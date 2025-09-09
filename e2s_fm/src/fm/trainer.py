# src/fm/trainer.py
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from tqdm import tqdm

from . import probability_path as pp  # import relativo

# -------------------------
# Base trainer (genérico)
# -------------------------
class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        ...

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad(set_to_none=True)
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f"Epoch {idx}, loss: {loss.item():.6f}")

        self.model.eval()
        return loss.detach()

# ------------------------------------------------------
# (Toy) Conditional Flow Matching con tensores planos
# Sigue siendo útil, pero sin depender de simple_model
# ------------------------------------------------------
class ConditionalFlowMatchingTrainer(Trainer):
    """
    Para datos de baja dimensión (x: [bs, dim]) y modelos MLP.
    """
    def __init__(self, path: pp.ConditionalProbabilityPath, model: torch.nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # z: (bs, dim)
        z = self.path.p_data.sample(batch_size)
        t = torch.rand(batch_size, 1).to(z)           # (bs, 1)
        x = self.path.sample_conditional_path(z, t)   # (bs, dim)

        ut_theta = self.model(x, t)                   # (bs, dim)
        ut_ref   = self.path.conditional_vector_field(x, z, t)  # (bs, dim)
        error = torch.sum(torch.square(ut_theta - ut_ref), dim=-1)  # (bs,)
        return torch.mean(error)

# ------------------------------------------------------
# (Toy) Conditional Score Matching con tensores planos
# ------------------------------------------------------
class ConditionalScoreMatchingTrainer(Trainer):
    """
    Para datos de baja dimensión (x: [bs, dim]) y modelos MLP.
    """
    def __init__(self, path: pp.ConditionalProbabilityPath, model: torch.nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size)             # (bs, dim)
        t = torch.rand(batch_size, 1).to(z)                 # (bs, 1)
        x = self.path.sample_conditional_path(z, t)         # (bs, dim)

        s_theta = self.model(x, t)                          # (bs, dim)
        s_ref   = self.path.conditional_score(x, z, t)      # (bs, dim)
        mse = torch.sum(torch.square(s_theta - s_ref), dim=-1)
        return torch.mean(mse)

# ------------------------------------------------------
# NUEVO: ImageCFMTrainer para imágenes (UNet)
# ------------------------------------------------------
class ImageCFMTrainer:
    """
    Conditional Flow Matching para IMÁGENES (x: [bs, C, H, W]).
    - path: objeto con .p_data.sample(bs) -> (x, y_dummy)
            .sample_conditional_path(z,t) y .conditional_vector_field(x,z,t)
    - model: UNet con forward(x, t, y) -> u_t^theta(x|y), mismo shape que x
    """
    def __init__(self, path: pp.ConditionalProbabilityPath, model: torch.nn.Module):
        self.path = path
        self.model = model

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, batch_size: int = 64):
        self.model.to(device).train()
        opt = self.get_optimizer(lr)

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            # 1) Muestreamos datos reales (z)
            z_samp = self.path.p_data.sample(batch_size)        # puede ser x o (x, y)
            # z: [bs, C, H, W] en [-1,1]
            if isinstance(z_samp, (tuple, list)):
                z = z_samp[0]
            else:
                z = z_samp

            z = z.to(device)

            bs, C, H, W = z.shape

            # 2) t ~ U(0,1) con broadcast
            t = torch.rand(batch_size, 1, 1, 1, device=device)  # [bs,1,1,1]

            # 3) x_t desde la probability path
            x = self.path.sample_conditional_path(z, t)         # [bs, C, H, W]

            # 4) Forward del UNet (unconditional → y=0)
            y = torch.zeros(batch_size, dtype=torch.long, device=device)
            ut_theta = self.model(x, t, y)                      # [bs, C, H, W]

            # 5) Vector field de referencia
            ut_ref = self.path.conditional_vector_field(x, z, t)# [bs, C, H, W]

            # 6) Pérdida per-pixel
            loss = (ut_theta - ut_ref).pow(2).mean()

            # 7) Optimización
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_description(f"epoch={epoch} loss={loss.item():.6f}")

        self.model.eval()
        return True
