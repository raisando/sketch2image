# src/fm/trainer.py
from __future__ import annotations
from abc import ABC, abstractmethod
import json
from pathlib import Path
from tqdm import tqdm
import math, copy
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from typing import Optional, Dict, List



from . import probability_path as pp


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
# ImageCFMTrainer REDONE
# ------------------------------------------------------
def ddp_is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

class ImageCFMTrainerLight:
    """
    Trainer mínimo para Flow Matching con UNet (DDP-aware):
      - usa train_loader
      - sin EMA, sin validación
      - history["train"] = promedio de loss por epoch (promedio GLOBAL entre GPUs)
      - con soporte opcional para AMP
    """
    def __init__(self, path, model, use_amp: bool = True):
        self.path = path
        self.model = model
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler('cuda',enabled=use_amp)

    @staticmethod
    def _get_x_from_batch(batch):
        if isinstance(batch, dict):
            return batch["x"] if "x" in batch else next(iter(batch.values()))
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch

    def get_train_loss(self, batch, device: torch.device) -> torch.Tensor:
        x = self._get_x_from_batch(batch).to(device, non_blocking=True)  # [B,C,H,W]
        B = x.shape[0]
        eps = 1e-3
        t = torch.rand(B, 1, 1, 1, device=device).clamp_(eps, 1.0 - eps)

        x_t   = self.path.sample_conditional_path(x, t)
        u_ref = self.path.conditional_vector_field(x_t, x, t)

        y = None
        if isinstance(batch, dict) and "y" in batch:
            y_in = batch["y"]
            if torch.is_tensor(y_in) and y_in.is_floating_point():
                y = y_in.to(device, non_blocking=True)

        u_pred = self.model(x_t, t, y)

        mse   = F.mse_loss(u_pred, u_ref)
        huber = F.smooth_l1_loss(u_pred, u_ref)
        return 0.5 * mse + 0.5 * huber

    def train(
        self,
        *,
        device: torch.device,
        lr: float = 1e-4,
        epochs: int = 200,
        train_loader,
        use_amp: bool = False,
    ) -> Dict[str, List[float]]:
        import time
        from torch.amp import autocast, GradScaler

        self.model.to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
        scaler = GradScaler('cuda',enabled=use_amp)

        history: Dict[str, List[float]] = {"train": []}

        bar_fmt_epoch = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} • {elapsed}<{remaining} • it/s:{rate_fmt}"
        bar_fmt_batch = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} • {elapsed}<{remaining}"

        if ddp_is_main():
            epoch_iter = tqdm(
                range(epochs),
                desc="epochs",
                ncols=90,
                dynamic_ncols=False,
                bar_format=bar_fmt_epoch,
            )
        else:
            epoch_iter = range(epochs)

        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        for epoch in epoch_iter:
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(int(epoch))

            self.model.train()
            epoch_loss_sum = 0.0
            epoch_sample_count = 0

            if ddp_is_main():
                batch_iter = tqdm(
                    train_loader,
                    desc=f"epoch {epoch+1}/{epochs}",
                    ncols=90,
                    dynamic_ncols=False,
                    bar_format=bar_fmt_batch,
                    leave=False,
                )
                t0 = time.time()
                imgs_seen_local = 0
            else:
                batch_iter = train_loader

            for batch in batch_iter:
                opt.zero_grad(set_to_none=True)

                if use_amp:
                    with autocast(device_type="cuda", enabled=True):
                        loss = self.get_train_loss(batch, device)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss = self.get_train_loss(batch, device)
                    loss.backward()
                    opt.step()

                bs = self._get_x_from_batch(batch).shape[0]
                epoch_loss_sum += float(loss.detach().item()) * bs
                epoch_sample_count += bs

                if ddp_is_main():
                    imgs_seen_local += bs
                    elapsed = max(1e-6, time.time() - t0)
                    it_s = batch_iter.n / elapsed              # iter/s local
                    img_s = (imgs_seen_local * world) / elapsed  # img/s global aprox
                    batch_iter.set_postfix(it_s=f"{it_s:.2f}", img_s=f"{img_s:.1f}", bs=f"{bs}x{world}")

            if ddp_is_main():
                batch_iter.close()

            # promedio global (todas las GPUs)
            loss_sum_tensor = torch.tensor([epoch_loss_sum, epoch_sample_count], device=device, dtype=torch.float64)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
            global_sum, global_count = loss_sum_tensor.tolist()
            epoch_loss = (global_sum / max(1.0, global_count))

            if ddp_is_main():
                history["train"].append(epoch_loss)

        return history if ddp_is_main() else {"train": []}
