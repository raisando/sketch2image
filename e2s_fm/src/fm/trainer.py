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
import math, copy, torch
import torch.nn.functional as F
from tqdm import tqdm

class ImageCFMTrainer(Trainer):
    def __init__(self, path, model, ema_decay=0.9999, use_amp=True, grad_clip=1.0):
        super().__init__(model)
        self.path = path
        self.ema_decay = ema_decay
        self.ema_model = None
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        if ema_decay is not None:
            try:
                self.ema_model = copy.deepcopy(self.model).eval()
                for p in self.ema_model.parameters():
                    p.requires_grad_(False)
            except Exception as e:
                print("[warn] EMA disabled:", e)
                self.ema_model = None

    def get_train_loss(self, **kwargs) -> torch.Tensor:
        # ya no se usa (satisface abstracto)
        raise NotImplementedError

    def _loss_from_batch(self, z: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        z: [B,C,H,W] ground-truth (muestra de p_data) en [-1,1]
        """
        z = z.to(device, non_blocking=True)
        B, C, H, W = z.shape
        eps = 1e-3
        t_img = torch.rand(B, 1, 1, 1, device=device).clamp(eps, 1.0 - eps)

        # x_t y u_ref desde el path (acepta imagen)
        x_t   = self.path.sample_conditional_path(z, t_img)          # [B,C,H,W]
        u_ref = self.path.conditional_vector_field(x_t, z, t_img)    # [B,C,H,W]

        y = torch.zeros(B, dtype=torch.long, device=device)  # uncond
        u_pred = self.model(x_t, t_img, y)

        # pérdida robusta
        mse   = F.mse_loss(u_pred, u_ref)
        huber = F.smooth_l1_loss(u_pred, u_ref)
        return 0.5 * mse + 0.5 * huber

    def _update_ema(self):
        if self.ema_model is None:
            return
        with torch.no_grad():
            for p, pe in zip(self.model.parameters(), self.ema_model.parameters()):
                pe.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3,
              train_loader=None, val_loader=None):
        assert train_loader is not None, "train_loader es requerido"
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)

        # scheduler warmup + cosine
        warmup = max(10, num_epochs // 20)
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            prog = (epoch - warmup) / max(1, (num_epochs - warmup))
            return 0.5 * (1 + math.cos(math.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and torch.cuda.is_available())

        history = {"train": [], "val": []}
        for epoch in tqdm(range(num_epochs), desc="train", leave=False, dynamic_ncols=True):
            # --------- TRAIN ----------
            self.model.train()
            running = 0.0
            for batch in train_loader:
                x = batch["x"]  # RightHalfImages debe exponer "x"
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp and torch.cuda.is_available()):
                    loss = self._loss_from_batch(x, device)
                scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                scaler.step(opt)
                scaler.update()
                self._update_ema()
                running += loss.item()
            epoch_train = running / max(1, len(train_loader))
            history["train"].append(epoch_train)

            # --------- VAL ----------
            if val_loader is not None:
                self.model.eval()
                val_acc = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch["x"]
                        with torch.cuda.amp.autocast(enabled=self.use_amp and torch.cuda.is_available()):
                            val_loss = self._loss_from_batch(x, device)
                        val_acc += val_loss.item()
                epoch_val = val_acc / max(1, len(val_loader))
                history["val"].append(epoch_val)

            sched.step()
            tqdm.write(f"epoch={epoch:04d} train={epoch_train:.4f}" + (f" val={epoch_val:.4f}" if val_loader else ""))

        return history
