import torch
from pathlib import Path
import argparse
from PIL import Image
import torchvision as tv

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from diffusers import DDPMScheduler, DDIMScheduler
from src.model.cond_unet import CondUNet
from src.tools.device import pick_device


def load_sketch_folder(sketch_dir, size=256, gray=True, limit=None):
    from torchvision import transforms as T
    p = Path(sketch_dir)
    files = sorted([*p.glob("*.png"), *p.glob("*.jpg")])
    if limit:
        files = files[:limit]
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    sks = []
    for f in files:
        im = Image.open(f).convert("RGB")
        im = tf(im)
        if gray:
            im = im.mean(dim=0, keepdim=True)
        sks.append(im*2-1)
    return torch.stack(sks, dim=0), files


@torch.no_grad()
def infer(model, scheduler, sketch, steps=50, use_ddim=False, eta=0.0, device="cuda"):
    sched = DDIMScheduler.from_config(scheduler.config) if use_ddim else DDPMScheduler.from_config(scheduler.config)
    sched.set_timesteps(steps)
    b = sketch.size(0)
    x = torch.randn(b, 3, sketch.size(2), sketch.size(3), device=device)
    for t in sched.timesteps:
        eps = model(x, t.repeat(b), sketch)
        x = sched.step(eps, t, x, eta=eta).prev_sample if use_ddim else sched.step(eps, t, x).prev_sample
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sketch_dir", required=True)
    ap.add_argument("--out", default="runs/min_ddpm/samples")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--gray_sketch", action="store_true")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--ddim", action="store_true")
    ap.add_argument("--eta", type=float, default=0.0)
    args = ap.parse_args()

    device = pick_device()
    print("Using device:", device)
    ckpt = torch.load(args.ckpt, map_location=device)
    in_ch = ckpt.get("in_ch", 4)
    size  = ckpt.get("size", args.size)

    model = CondUNet(in_channels_total=in_ch, out_channels=3, sample_size=size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    # carga sketches sueltos
    sk, files = load_sketch_folder(args.sketch_dir, size=size, gray=args.gray_sketch)
    sk = sk.to(device)

    imgs = infer(model, scheduler, sk, steps=args.steps, use_ddim=args.ddim, eta=args.eta, device=device)
    imgs = (imgs.clamp(-1,1) + 1) * 0.5

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    for i, (im, f) in enumerate(zip(imgs, files)):
        tv.utils.save_image(im, out_dir / f"gen_{Path(f).stem}.png")
    print(f"[OK] Guardadas {len(files)} imágenes en {out_dir}")

if __name__ == "__main__":
    main()
