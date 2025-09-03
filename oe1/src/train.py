import os
# Disable SDPA kernels so diffusers won't hit the MPS path that crashes
os.environ.setdefault("PYTORCH_ENABLE_FLASH_SDP", "0")
os.environ.setdefault("PYTORCH_ENABLE_MEM_EFFICIENT_SDP", "0")
os.environ.setdefault("PYTORCH_ENABLE_MATH_SDP", "1")
# Let PyTorch bounce specific ops to CPU if necessary on Apple GPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

# ---- Safe attention for MPS (avoid SDPA kernel) ----
if torch.backends.mps.is_available():
    def _sdpa_safe(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        # q,k,v: (..., L, d), (..., S, d), (..., S, d)
        d = q.shape[-1]
        if scale is None:
            scale = 1.0 / (d ** 0.5)
        # matmul attention (float32 for stability)
        qf = q.float()
        kf = k.float()
        vf = v.float()
        attn = (qf @ kf.transpose(-2, -1)) * scale
        if is_causal:
            L, S = attn.shape[-2], attn.shape[-1]
            causal = torch.ones((L, S), device=attn.device, dtype=torch.bool).triu(1)
            attn = attn.masked_fill(causal, float('-inf'))
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        # (optional) dropout on attn in training; skipped for simplicity
        out = attn @ vf
        # cast back
        return out.to(q.dtype)

    F.scaled_dot_product_attention = _sdpa_safe
# -----------------------------------------------------
from torch.optim import AdamW
from tqdm import tqdm
from src.tools.device import pick_device
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from diffusers import DDPMScheduler
from src.datasets.paired_pix2pix import make_loaders
from src.model.cond_unet import CondUNet
from src.utils import save_grid, sample_batch, eval_val_loss
from torch.utils.tensorboard import SummaryWriter



def main():
    print(">> START train")  # debug mínimo
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
    p.add_argument("--epochs", type=int, default=None)           # optional, converts to steps
    p.add_argument("--eval_every", type=int, default=500)        # how often to run val
    p.add_argument("--patience", type=int, default=5)            # early-stopping patience (eval windows)
    p.add_argument("--min_delta", type=float, default=1e-4)      # required improvement to reset patience

    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    writer = SummaryWriter(log_dir=args.out)   # logs go in your run folder

    device = pick_device()
    print("Using device:", device)
    os.makedirs(args.out, exist_ok=True)

    # loaders
    train_loader, val_loader, test_loader = make_loaders(
        data_root=args.data_root, size=args.size, batch_size=args.batch, num_workers=4, aug=True, gray_sketch=args.gray_sketch
    )

    # modelo (in_channels: 3 + (1 si gray_sketch else 3))
    in_ch = 3 + (1 if args.gray_sketch else 3)
    model = CondUNet(in_channels_total=in_ch, out_channels=3, sample_size=args.size).to(device)
    print("UNet in_channels:", getattr(getattr(model, "unet", None), "config", {}).in_channels)

    # 🔧 MPS attention workaround (newer diffusers) + fallback for older versiones
    try:
        from diffusers.models.attention_processor import AttnProcessor
        if hasattr(model, "unet") and hasattr(model.unet, "set_attn_processor"):
            model.unet.set_attn_processor(AttnProcessor())
            print("[info] Using AttnProcessor() (SDPA disabled inside diffusers UNet)")
        else:
            print("[warn] UNet.set_attn_processor not found; relying on global SDPA disable env vars.")
    except Exception as e:
        print("[warn] Could not set AttnProcessor:", e)
        print("[warn] Relying on global SDPA disable env vars.")

    # scheduler DDPM (cosine)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    opt = AdamW(model.parameters(), lr=args.lr)

    step = 0
    model.train()
    pbar = tqdm(total=args.steps, desc="train")

    batches_per_epoch = (len(train_loader.dataset) + args.batch - 1) // args.batch
    if args.epochs is not None:
        args.steps = args.epochs * batches_per_epoch
    print(f"[info] steps={args.steps} (~{batches_per_epoch} batches/epoch)")

    best_val = float("inf")
    since_improve = 0

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

            if step == 0:
                print("x_t:", tuple(x_t.shape), x_t.dtype, x_t.device)
                print("sk :", tuple(sk.shape),  sk.dtype,  sk.device)
                print("t  :", tuple(t.shape),   t.dtype,   t.device)

            t = t.to(x_t.device)
            # Some setups prefer float timesteps; this avoids SDPA dtype quirks
            t_for_unet = t.to(dtype=x_t.dtype)

            eps_pred = model(x_t, t_for_unet, sk)
            loss = F.mse_loss(eps_pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            writer.add_scalar("train/loss", loss.item(), step)  # 👈 here

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            pbar.update(1)

            if step % args.save_every == 0:
                model.eval()
                with torch.no_grad():
                    # intenta obtener un batch de val; si no hay, usa train
                    try:
                        vb = next(iter(val_loader))
                    except Exception:
                        vb = next(iter(train_loader))
                    skv = vb["sketch"].to(device)
                    if args.gray_sketch and skv.shape[1] == 3:
                        skv = skv.mean(dim=1, keepdim=True)

                    imgs = sample_batch(model, scheduler, skv[:8], steps=50, device=device)
                    save_grid(imgs, Path(args.out)/f"sample_step{step}.png")

                # guarda checkpoint intermedio
                ckpt_dir = Path(args.out) / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(),
                            "size": args.size,
                            "in_ch": in_ch,
                            "step": step},
                            ckpt_dir / f"step_{step:06d}.pt")

                model.train()

            # ----- periodic validation & early stopping -----
            if step % args.eval_every == 0:
                val_loss = eval_val_loss(model, val_loader, scheduler, device, args.gray_sketch)
                writer.add_scalar("val/loss", val_loss, step)  # optional TB log
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "val": f"{val_loss:.4f}"})

                ckpt_dir = Path(args.out) / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                # Save best
                if val_loss + args.min_delta < best_val:
                    best_val = val_loss
                    since_improve = 0
                    torch.save({"model": model.state_dict(),
                                "size": args.size,
                                "in_ch": in_ch,
                                "step": step,
                                "val_loss": val_loss},
                            ckpt_dir / "best.pt")
                    # print or pbar.write so tqdm isn’t broken:
                    pbar.write(f"[best] step={step} val={val_loss:.4f}")
                else:
                    since_improve += 1
                    pbar.write(f"[no-improve {since_improve}/{args.patience}] step={step} val={val_loss:.4f} best={best_val:.4f}")

                # Early stop?
                if since_improve >= args.patience:
                    pbar.write(f"[early-stop] no improvement in {args.patience} evals (best={best_val:.4f}).")
                    # Optionally load best back to RAM before exiting:
                    # state = torch.load(ckpt_dir / "best.pt", map_location=device)
                    # model.load_state_dict(state["model"])
                    # break both loops:
                    step = args.steps  # force outer condition to stop
                    break

            if step >= args.steps:
                break
    # guarda pesos y config mínima
    torch.save({"model": model.state_dict(), "size": args.size, "in_ch": in_ch}, Path(args.out)/"ckpt.pt")
    print(f"[OK] Entrenamiento terminado. Checkpoints en {args.out}")

if __name__ == "__main__":
    main()
