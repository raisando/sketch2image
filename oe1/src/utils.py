# src/utils.py
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision as tv
from diffusers import DDPMScheduler

def save_grid(x: torch.Tensor, path: Path, nrow: int = 4) -> None:
    x = (x.clamp(-1, 1) + 1) * 0.5  # [-1,1] -> [0,1]
    tv.utils.save_image(x, path, nrow=nrow)

@torch.no_grad()
def sample_batch(model, scheduler: DDPMScheduler, cond: torch.Tensor, steps: int, device: str):
    model.eval()
    scheduler.set_timesteps(steps, device=device)
    b, _, h, w = cond.shape
    x = torch.randn((b, 3, h, w), device=device)
    for t in scheduler.timesteps:
        eps = model(x, t.to(dtype=x.dtype, device=x.device), cond)
        out = scheduler.step(model_output=eps, timestep=t, sample=x)
        x = out.prev_sample
    return x

@torch.no_grad()
def eval_val_loss(model, loader, scheduler: DDPMScheduler, device: str, gray_sketch: bool) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        x0 = batch["photo"].to(device).float()
        sk = batch["sketch"].to(device).float()
        if gray_sketch and sk.shape[1] == 3:
            sk = sk.mean(dim=1, keepdim=True)

        bsz = x0.size(0)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device)
        noise = torch.randn_like(x0)
        x_t = scheduler.add_noise(x0, noise, t)
        t_for_unet = t.to(dtype=x_t.dtype)

        eps = model(x_t, t_for_unet, sk)
        loss = F.mse_loss(eps, noise, reduction="mean")
        total += loss.item() * bsz
        count += bsz
    return total / max(count, 1)
