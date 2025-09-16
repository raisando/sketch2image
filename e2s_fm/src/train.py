# src/train.py
import argparse, torch, json, datetime, copy
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from src.datasets.right_half_only import make_loaders, ImageDatasetSampler, RightHalfImages
from src.fm import alpha_beta as ab, probability_path as pp
from src.fm.trainer import ImageCFMTrainer
from src.model.unet import FMUNet
from src.tools.utils import sample_and_save, eval_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default="runs/fm_e2s_uncond")
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"[info] device={device}")

    # Loaders
    train_loader, val_loader, test_loader = make_loaders(
        args.data_root, size=args.size, batch_size=args.batch, to_gray=args.gray,
        num_workers=args.num_workers
    )

    # p_data solo para path API (no lo usaremos para samplear batches ya)
    p_train = ImageDatasetSampler(RightHalfImages(args.data_root, split="train", size=args.size, to_gray=args.gray)).to(device)

    # Path (para x_t y u_ref)
    path = pp.GaussianConditionalProbabilityPath(
        p_data=p_train,
        alpha=ab.LinearAlpha(),
        beta=ab.SquareRootBeta()
    ).to(device)

    C = 1 if args.gray else 3
    channels = [32, 64, 128]
    depth = 2

    unet = FMUNet(
        channels=channels,
        num_residual_layers=depth,
        t_embed_dim=40, y_embed_dim=40,
        in_channels=C, out_channels=C, num_classes=1
    ).to(device)

    # Trainer con AMP/EMA
    trainer = ImageCFMTrainer(path, unet, ema_decay=0.9999, use_amp=True)
    print("[info] start training")

    history = trainer.train(
        num_epochs=args.epochs, device=device, lr=args.lr,
        train_loader=train_loader, val_loader=val_loader
    )

    outdir = Path(args.out)
    (outdir).mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print("[ok] saved:", metrics_path)

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(history["train"], label="train")
    if history["val"]:
        plt.plot(history["val"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Flow Matching - entrenamiento")
    plt.legend(loc="upper right")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    hparams = (
        f"res={args.size}, C={C}\n"
        f"channels={channels}, depth={depth}\n"
        f"batch={args.batch}, lr={args.lr}\n"
        f"epochs={args.epochs}, device={device}\n"
        f"{stamp}"
    )
    plt.gcf().text(0.98, 0.02, hparams, ha="right", va="bottom",
                   bbox=dict(boxstyle="round", fc="white", ec="gray"))
    diag_path = outdir / "diagnostics.png"
    plt.tight_layout(); plt.savefig(diag_path, dpi=150); plt.close()
    print("[ok] saved:", diag_path)

    # Pesos
    torch.save(unet.state_dict(), outdir / "model.pth")
    print("[ok] saved:", outdir / "model.pth")
    if getattr(trainer, "ema_model", None) is not None:
        torch.save(trainer.ema_model.state_dict(), outdir / "model_ema.pth")
        print("[ok] saved:", outdir / "model_ema.pth")

    # Test loss con EMA (opcional)
    p_test = ImageDatasetSampler(RightHalfImages(args.data_root, split="test", size=args.size, to_gray=args.gray)).to(device)
    test_loss = eval_loss(trainer, p_test, device, batches=min(100, len(test_loader) or 100), batch_size=args.batch, use_ema=True)
    print(f"[report] test_loss (EMA): {test_loss:.6f}")

    # Samples (EMA si existe)
    samples_path = outdir / "samples.png"
    model_for_sampling = unet
    if (outdir / "model_ema.pth").exists():
        ema = FMUNet(channels=channels, num_residual_layers=depth, t_embed_dim=40, y_embed_dim=40,
                     in_channels=C, out_channels=C, num_classes=1).to(device)
        ema.load_state_dict(torch.load(outdir / "model_ema.pth", map_location=device))
        model_for_sampling = ema

    sample_and_save(model_for_sampling, samples_path, num=36, size=args.size, channels=C, steps=750, device=device)
    print("[ok] saved:", samples_path)

if __name__ == "__main__":
    main()
