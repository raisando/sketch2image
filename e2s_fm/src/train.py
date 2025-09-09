import argparse, torch
from pathlib import Path
from src.datasets.right_half_only import RightHalfImages, ImageDatasetSampler
from src.fm import alpha_beta as ab
from src.fm import probability_path as pp
from src.fm.trainer import ImageCFMTrainer
from src.model.unet import FMUNet
from src.fm import distrib_utils as distrib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="runs/fm_e2s_uncond")
    ap.add_argument("--gray", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"[info] device={device}")

    # Dataset + sampler (uncond)
    dataset_sampler = RightHalfImages(args.data_root, split="train", size=args.size, to_gray=args.gray)
    p_data = ImageDatasetSampler(dataset_sampler).to(device)

    C = 1 if args.gray else 3

    # Probability path
    path = pp.GaussianConditionalProbabilityPath(
        p_data=p_data,
        alpha=ab.LinearAlpha(),
        beta=ab.SquareRootBeta()
    ).to(device)

    unet = FMUNet(
        channels=[32,64,128],
        num_residual_layers=2,
        t_embed_dim=40,
        y_embed_dim=40,
        in_channels=C,      # <- 1 (gris) o 3 (RGB)
        out_channels=C,     # <- genera misma cantidad de canales
        num_classes=1       # <- unconditional
    ).to(device)

    # trainer nuevo
    trainer = ImageCFMTrainer(path, unet)

    print("[info] start training")
    losses = trainer.train(
        num_epochs=args.epochs,
        device=device,
        lr=args.lr,
        batch_size=args.batch
    )

    # Guardar
    torch.save(unet.state_dict(), Path(args.out)/"edges2shoes_uncond.pth")
    print("[ok] saved:", Path(args.out)/"edges2shoes_uncond.pth")


if __name__ == "__main__":
    main()
