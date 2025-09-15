# src/fm/trainer.py
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
import math, copy
import torch
import torch.nn.functional as F
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
class ImageCFMTrainer(Trainer):
    def __init__(self, path, model, ema_decay=0.999):
        super().__init__(model)
        self.path = path
        self.ema_decay = ema_decay
        self.ema_model = None
        # Intentar configurar EMA, pero no morir si falla
        if ema_decay is not None:
            try:
                self.ema_model = copy.deepcopy(self.model).eval()
                for p in self.ema_model.parameters():
                    p.requires_grad_(False)
            except Exception as e:
                print("[warn] EMA disabled:", e)
                self.ema_model = None

    def get_train_loss(self, **kwargs) -> torch.Tensor:
        batch_size = kwargs.get("batch_size", 64)
        device = kwargs.get(
            "device",
            next(self.model.parameters()).device
        )
        return self._one_batch_loss(batch_size, device)

    def _one_batch_loss(self, batch_size: int, device: torch.device):
        # z ~ p_data
        z_samp = self.path.p_data.sample(batch_size)
        z = z_samp[0] if isinstance(z_samp, (tuple, list)) else z_samp   # [B,C,H,W]
        z = z.to(device)
        B, C, H, W = z.shape

        eps = 1e-3
        t_img = torch.rand(B, 1, 1, 1, device=device).clamp(eps, 1.0 - eps)
        t_vec = t_img.view(B, 1)

        # x_t y u_ref desde el path (soporta imagen o vector)
        try:
            x_t   = self.path.sample_conditional_path(z, t_img)
            u_ref = self.path.conditional_vector_field(x_t, z, t_img)
        except Exception:
            z_vec   = z.view(B, -1)
            x_vec   = self.path.sample_conditional_path(z_vec, t_vec)
            u_ref_v = self.path.conditional_vector_field(x_vec, z_vec, t_vec)
            x_t   = x_vec.view(B, C, H, W)
            u_ref = u_ref_v.view(B, C, H, W)

        y = torch.zeros(B, dtype=torch.long, device=device)  # uncond
        u_pred = self.model(x_t, t_img, y)

        # pérdida robusta
        mse   = F.mse_loss(u_pred, u_ref)
        huber = F.smooth_l1_loss(u_pred, u_ref)
        loss  = 0.5 * mse + 0.5 * huber
        return loss

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, batch_size: int = 64,
              val_p_data=None, val_batches: int = 5):
        self.model.to(device)
        self.ema_model.to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)

        # scheduler con warmup + cosine
        warmup = max(10, num_epochs // 20)
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            prog = (epoch - warmup) / max(1, (num_epochs - warmup))
            return 0.5 * (1 + math.cos(math.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        # opcional: path de validación (mismo path pero p_data=val)
        path_val = None
        if val_p_data is not None:
            # clonar path con p_data de validación (simple, alpha, beta iguales)
            path_val = copy.copy(self.path)
            path_val.p_data = val_p_data.to(device)

        train_losses, val_losses = [], []

        pbar = tqdm(range(num_epochs), desc="train")
        for epoch in pbar:
            self.model.train()
            opt.zero_grad(set_to_none=True)
            loss = self._one_batch_loss(batch_size, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            # EMA (opcional)
            if self.ema_model is not None:
                with torch.no_grad():
                    for p, pe in zip(self.model.parameters(), self.ema_model.parameters()):
                        pe.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)


            sched.step()
            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{opt.param_groups[0]['lr']:.2e}"})

            # validación liviana (promedio de N batches)
            if path_val is not None:
                self.model.eval()
                with torch.no_grad():
                    acc = 0.0
                    for _ in range(val_batches):
                        # usar temporalmente el path_val
                        old = self.path
                        self.path = path_val
                        val_loss = self._one_batch_loss(batch_size, device)
                        self.path = old
                        acc += val_loss.item()
                    val_losses.append(acc / val_batches)

        return {"train": train_losses, "val": val_losses}
