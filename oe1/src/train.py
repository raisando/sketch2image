import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from diffusers import DDPMScheduler
from src.datasets.paired_pix2pix import make_loaders
from src.model.cond_unet import CondUNet

def save_grid(x, path):
    import torchvision as tv
    x = (x.clamp(-1,1) + 1) * 0.5
    tv.utils.save_image(x, path, nrow=4)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--out", type=str, default="runs/min_ddpm")
    p.add_argument("--gray_sketch", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    # loaders
    train_loader, val_loader, test_loader = make_loaders(
        data_root=args.data_root, size=args.size, batch_size=args.batch, num_workers=4, aug=True, gray_sketch=args.gray_sketch
    )

    # modelo (in_channels: 3 + (1 si gray_sketch else 3))
    in_ch = 3 + (1 if args.gray_sketch else 3)
    model = CondUNet(in_channels_total=in_ch, out_channels=3, sample_size=args.size).to(device)

    # scheduler DDPM (cosine)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    opt = AdamW(model.parameters(), lr=args.lr)

    step = 0
    model.train()
    pbar = tqdm(total=args.steps, desc="train")

    while step < args.steps:
        for batch in train_loader:
            x0 = batch["photo"].to(device)
            sk = batch["sketch"].to(device)
            # si sketch es [3,H,W] y pusiste --gray_sketch, promedia a 1 canal
            if args.gray_sketch and sk.shape[1] == 3:
                sk = sk.mean(dim=1, keepdim=True)

            bsz = x0.size(0)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device)
            noise = torch.randn_like(x0)
            x_t = scheduler.add_noise(x0, noise, t)

            eps_pred = model(x_t, t, sk)
            loss = F.mse_loss(eps_pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            pbar.update(1)

            if step % args.save_every == 0:
                # muestra rápido 1 batch del val con DDPM 50 pasos
                model.eval()
                with torch.no_grad():
                    try:
                        vb = next(iter(val_loader))
                    except StopIteration:
                        vb = next(iter(val_loader))
                    skv = vb["sketch"].to(device)
                    if args.gray_sketch and skv.shape[1] == 3:
                        skv = skv.mean(dim=1, keepdim=True)
                    imgs = sample_batch(model, scheduler, skv[:8], steps=50, device=device)
                    save_grid(imgs, Path(args.out)/f"sample_step{step}.png")
                model.train()

            if step >= args.steps:
                break


    # guarda pesos y config mínima
    torch.save({"model": model.state_dict(), "size": args.size, "in_ch": in_ch}, Path(args.out)/"ckpt.pt")
    print(f"[OK] Entrenamiento terminado. Checkpoints en {args.out}")
