# sample.py
import argparse, torch
from pathlib import Path
from src.model.unet import FMUNet
from src.tools.utils import sample_and_save

import torch.nn as nn
class NegatedModel(nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner
    def forward(self, x, t, y=None): return -self.inner(x, t, y)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="ruta al checkpoint .pth (model o model_ema)")
    ap.add_argument("--out", type=str, default="samples.png")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--num", type=int, default=36)
    ap.add_argument("--steps", type=int, default=750)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # definir tu UNet con los mismos hparams que en train.py
    unet = FMUNet(
        channels=[32,64,128],
        num_residual_layers=2,
        t_embed_dim=40, y_embed_dim=40,
        in_channels=args.channels, out_channels=args.channels,
        num_classes=1
    ).to(device)

    # cargar pesos
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    unet.load_state_dict(state)
    print(f"[ok] cargado {args.ckpt}")

    model_for_sampling = NegatedModel(unet).to(device)

    # samplear
    sample_and_save(model_for_sampling, Path(args.out),
                    num=args.num, size=args.size, channels=args.channels,
                    steps=args.steps, device=device)
    print(f"[ok] guardado {args.out}")

if __name__ == "__main__":
    main()
