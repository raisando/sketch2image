# src/fm/trainer.py
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
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
# ImageCFMTrainer (readable, step-based, safer AMP)
# ------------------------------------------------------
import json, math, copy, torch
from pathlib import Path
from typing import Optional, Dict, List
import torch.nn.functional as F
from tqdm import tqdm

class ImageCFMTrainer(Trainer):
    """
    Trainer for conditional flow matching with image UNets.
    """

    def __init__(
        self,
        path,
        model,
        ema_decay: Optional[float] = 0.9999,
        use_amp: bool = True,
        grad_clip: Optional[float] = 1.0,
    ):
        super().__init__(model)
        self.path = path
        self.ema_decay = ema_decay
        self.use_amp = use_amp
        self.grad_clip = grad_clip

        self.ema_model = None
        if ema_decay is not None:
            try:
                self.ema_model = copy.deepcopy(self.model).eval()
                for p in self.ema_model.parameters():
                    p.requires_grad_(False)
            except Exception as e:
                print("[warn] EMA disabled:", e)
                self.ema_model = None

        self.save_dir: Optional[Path] = None

    def get_train_loss(self, **kwargs) -> torch.Tensor:
        """
        Implementación concreta para satisfacer la ABC.
        No se usa en el loop por-steps, pero permite instanciar sin error.
        """
        device = kwargs.get("device", next(self.model.parameters()).device)
        batch_size = kwargs.get("batch_size", 64)
        return self._one_batch_loss(batch_size=batch_size, device=device)

    # === Core losses ===
    def _loss_from_batch(self, z: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        z: [B,C,H,W] ground-truth (sampled from p_data) in [-1, 1].
        """
        z = z.to(device, non_blocking=True)
        B, C, H, W = z.shape

        eps = 1e-3
        t_img = torch.rand(B, 1, 1, 1, device=device).clamp_(eps, 1.0 - eps)

        x_t   = self.path.sample_conditional_path(z, t_img)       # [B,C,H,W]
        u_ref = self.path.conditional_vector_field(x_t, z, t_img) # [B,C,H,W]

        y = torch.zeros(B, dtype=torch.long, device=device)
        u_pred = self.model(x_t, t_img, y)

        mse   = F.mse_loss(u_pred, u_ref)
        huber = F.smooth_l1_loss(u_pred, u_ref)
        return 0.5 * mse + 0.5 * huber

    # === EMA ===
    def _update_ema(self):
        if self.ema_model is None:
            return
        with torch.no_grad():
            d = self.ema_decay
            for p, pe in zip(self.model.parameters(), self.ema_model.parameters()):
                pe.data.mul_(d).add_(p.data, alpha=1.0 - d)

    # === Validation / Checkpoint / Save Metrics ===
    @torch.no_grad()
    def _validate(
        self,
        device: torch.device,
        val_loader,
        amp_enabled: bool,
        device_type: str,
    ) -> float:
        self.model.eval()
        tot = 0.0
        for batch in val_loader:
            x = batch["x"]
            with torch.amp.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=amp_enabled
            ):
                val_loss = self._loss_from_batch(x, device)
            tot += float(val_loss.item())
        return tot / max(1, len(val_loader))

    def _save_checkpoint(
        self,
        step: int,
        opt: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LRScheduler,
        best_val: float,
    ) -> Path:
        ckpt_dir = (self.save_dir or Path(".")) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"step_{step}.pth"
        torch.save({
            "step": step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict() if self.ema_model is not None else None,
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "best_val": best_val,
        }, ckpt_path)
        return ckpt_path

    def _save_best(self):
        if self.save_dir is None:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_dir / "model_best.pth")
        if self.ema_model is not None:
            torch.save(self.ema_model.state_dict(), self.save_dir / "model_ema_best.pth")

    def _write_metrics(self, history: Dict[str, List[float]]):
        if self.save_dir is None:
            return
        try:
            path = self.save_dir / "metrics.json"
            with path.open("w") as f:
                json.dump(history, f)
        except Exception as e:
            print(f"[warn] could not write metrics.json: {e}")

    # === Training ===
    def train(
        self,
        device       : torch.device,
        lr           : float = 1e-3,
        train_loader=None,
        val_loader=None,
        max_updates  : int = 120_000,
        eval_every   : int = 5_000,
        patience     : int = 3,
        num_epochs   : Optional[int] = None,
        save_dir     : Optional[str] = None,
        warmup_ratio : float = 0.025,
    ):
        """
        Step-based training (recommended). If you must use epoch mode, wrap externally.

        Args:
            device: torch.device
            lr: base learning rate
            train_loader: yields dict with 'x' = images in [-1,1]
            val_loader:   same format; enables early stopping if provided
                max_updates:  total optimizer steps
                eval_every:   validate & checkpoint every N steps
                patience:     early-stop after N validations without improvement
                save_dir:     directory to write best weights & metrics.json
                warmup_ratio: fraction of steps used for linear warmup
        """
        assert train_loader is not None, "train_loader is required"
        self.save_dir = Path(save_dir) if save_dir is not None else None

        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)

        # Optimizer
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.99),
            weight_decay=1e-4
        )

        # LR schedule
        total_steps = max(1, max_updates) if (max_updates and max_updates > 0) else 1
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        after_warmup = max(1, total_steps - warmup_steps)

        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0/warmup_steps, total_iters=warmup_steps) \
                 if warmup_steps > 0 else torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=after_warmup)

        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        # AMP
        device_type = device.type
        amp_enabled = self.use_amp and device_type in {"cuda", "mps"}
        scaler = torch.amp.GradScaler(device="cuda") if (device_type == "cuda" and amp_enabled) else None

        # Checkpoint
        history: Dict[str, List[float]] = {"train": [], "val": []}
        best_val = float("inf")
        best_state = None
        best_ema_state = None
        validations_since_best = 0

        # Loop state
        epoch = 0
        running_loss = 0.0
        loader_iter = iter(train_loader)

        pbar = tqdm(total=total_steps, dynamic_ncols=True, desc="train(steps)")

        for step in range(1, total_steps + 1):
            try:
                batch = next(loader_iter)
            except StopIteration:
                epoch += 1
                if len(train_loader) > 0:
                    history["train"].append(running_loss / len(train_loader))
                running_loss = 0.0
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            self.model.train()
            x = batch["x"]
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=amp_enabled
            ):
                loss = self._loss_from_batch(x, device)

            if scaler is not None:
                scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                opt.step()

            sched.step()
            self._update_ema()

            # Logging
            running_loss += float(loss.item())
            current_lr = opt.param_groups[0]["lr"]
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

            # Validation / checkpoint / early stopping
            eval_step = (val_loader is not None) and (step % eval_every == 0 or step == total_steps)
            if eval_step:
                train_avg = (running_loss / len(train_loader)) if len(train_loader) > 0 else float("nan")
                history["train"].append(train_avg)
                running_loss = 0.0

                val_loss = self._validate(device, val_loader, amp_enabled, device_type)
                history["val"].append(val_loss)
                tqdm.write(f"[val] step={step} val={val_loss:.4f}")

                # Checkpoint snapshot
                ckpt_path = self._save_checkpoint(step, opt, sched, best_val)
                tqdm.write(f"[ckpt] saved {ckpt_path}")

                # Best tracking
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    validations_since_best = 0
                    best_state = copy.deepcopy(self.model.state_dict())
                    if self.ema_model is not None:
                        best_ema_state = copy.deepcopy(self.ema_model.state_dict())
                    self._save_best()
                    tqdm.write(f"[best] improved to {best_val:.4f} (saved best weights)")
                else:
                    validations_since_best += 1
                    if validations_since_best >= patience:
                        tqdm.write(f"[early-stop] no improvement for {patience} validations.")
                        break

                # Persist metrics incrementally so they survive crashes
                self._write_metrics(history)

        pbar.close()

        # Restore best states (so the returned model is the best)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if self.ema_model is not None and best_ema_state is not None:
            self.ema_model.load_state_dict(best_ema_state)

        # Final write
        self._write_metrics(history)
        return history


class ImageCFMTrainerEpoch(ImageCFMTrainer):
    """
    Entrenamiento por epochs: valida al final de cada epoch y guarda checkpoints epoch_k.pth.
    """
    def train(
        self,
        device       : torch.device,
        lr           : float = 1e-3,
        train_loader=None,
        val_loader=None,
        num_epochs   : int = 200,
        patience     : int = 999999,
        save_dir     : Optional[str] = None,
        warmup_ratio : float = 0.025,
    ):
        assert train_loader is not None, "train_loader is required"
        self.save_dir = Path(save_dir) if save_dir is not None else None

        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)

        # Optimizer
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4
        )

        # LR schedule (warmup + cosine por-batch; simple y funciona bien)
        total_steps = max(1, num_epochs * max(1, len(train_loader)))
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        after_warmup = max(1, total_steps - warmup_steps)
        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0/warmup_steps, total_iters=warmup_steps) \
            if warmup_steps > 0 else torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=after_warmup)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[warmup_steps])

        # AMP
        device_type = device.type
        amp_enabled = self.use_amp and device_type in {"cuda", "mps"}
        scaler = torch.amp.GradScaler(device="cuda") if (device_type == "cuda" and amp_enabled) else None

        # Estado
        history: Dict[str, List[float]] = {"train": [], "val": []}
        best_val = float("inf")
        best_state = None
        best_ema_state = None
        no_improve = 0

        for epoch in tqdm(range(num_epochs), desc="train(epochs)", dynamic_ncols=True):
            self.model.train()
            running = 0.0

            for batch in train_loader:
                x = batch["x"]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_enabled):
                    loss = self._loss_from_batch(x, device)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    opt.step()

                sched.step()
                self._update_ema()
                running += float(loss.item())

            # fin de epoch: log y validación
            epoch_train = running / max(1, len(train_loader))
            history["train"].append(epoch_train)

            if val_loader is not None:
                val_loss = self._validate(device, val_loader, amp_enabled, device_type)
                history["val"].append(val_loss)
                tqdm.write(f"[val] epoch={epoch} val={val_loss:.4f}")

                # checkpoint por epoch
                if self.save_dir is not None:
                    ckpt_path = self.save_dir / "checkpoints" / f"epoch_{epoch}.pth"
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "ema": self.ema_model.state_dict() if self.ema_model is not None else None,
                        "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(),
                        "best_val": best_val,
                    }, ckpt_path)
                    tqdm.write(f"[ckpt] saved {ckpt_path}")

                # best / early-stop
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    no_improve = 0
                    best_state = copy.deepcopy(self.model.state_dict())
                    if self.ema_model is not None:
                        best_ema_state = copy.deepcopy(self.ema_model.state_dict())
                    self._save_best()
                    tqdm.write(f"[best] improved to {best_val:.4f}")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        tqdm.write(f"[early-stop] no improvement for {patience} epochs.")
                        break

                self._write_metrics(history)

        # Restaurar best
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if self.ema_model is not None and best_ema_state is not None:
            self.ema_model.load_state_dict(best_ema_state)

        self._write_metrics(history)
        return history
