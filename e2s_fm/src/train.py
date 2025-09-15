# al inicio
import argparse
import copy, json, datetime
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torchvision.utils import make_grid, save_image

from src.datasets.right_half_only import RightHalfImages, ImageDatasetSampler
from src.fm import alpha_beta as ab, probability_path as pp
from src.fm.trainer import ImageCFMTrainer
from src.model.unet import FMUNet
from src.tools.utils import eval_loss, sample_and_save   # <-- NUEVO

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

    C = 1 if args.gray else 3

    p_data  = ImageDatasetSampler(RightHalfImages(args.data_root, split="train", size=args.size, to_gray=args.gray)).to(device)
    p_data_val = ImageDatasetSampler(RightHalfImages(args.data_root, split="val",   size=args.size, to_gray=args.gray)).to(device)
    p_test = ImageDatasetSampler(RightHalfImages(args.data_root, split="test", size=args.size, to_gray=args.gray)).to(device)


    # canales internos parametrizables (opcional)
    # channels = [32, 64, 128]  # fijo
    channels = [32, 64, 128]  # deja esto como quieras (o hazlo flag)
    depth = 2                 # Residuales por bloque (R=2)

    # UNet
    C = 1 if args.gray else 3
    unet = FMUNet(
        channels=channels,
        num_residual_layers=depth,
        t_embed_dim=40, y_embed_dim=40,
        in_channels=C, out_channels=C, num_classes=1
    ).to(device)

    # Path
    path = pp.GaussianConditionalProbabilityPath(
        p_data=p_data,
        alpha=ab.LinearAlpha(),
        beta=ab.SquareRootBeta()
    ).to(device)

    # Train con validación
    trainer = ImageCFMTrainer(path, unet)
    print("[info] start training")
    history = trainer.train(
        num_epochs=args.epochs, device=device, lr=args.lr, batch_size=args.batch,
        val_p_data=p_data_val, val_batches=5
    )

    test_loss = eval_loss(trainer, p_test, device, batches=50, batch_size=args.batch, use_ema=True)
    print(f"[report] test_loss (EMA): {test_loss:.6f}")
    history["test_final"] = test_loss

    # --- guardar métricas ---
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print("[ok] saved:", metrics_path)

    # --- gráfico de diagnóstico ---
    plt.figure(figsize=(8,5))
    plt.plot(history["train"], label="train")
    if history["val"]:
        plt.plot(history["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Flow Matching - entrenamiento")
    plt.legend(loc="upper right")

    # cuadro de hiperparámetros
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
    plt.tight_layout()
    plt.savefig(diag_path, dpi=150)
    plt.close()
    print("[ok] saved:", diag_path)

    # --- guardar pesos (modelo y EMA) ---
    torch.save(unet.state_dict(), outdir / "model.pth")
    print("[ok] saved:", outdir / "model.pth")
    if hasattr(trainer, "ema_model"):
        torch.save(trainer.ema_model.state_dict(), outdir / "model_ema.pth")
        print("[ok] saved:", outdir / "model_ema.pth")

    # --- sampleo de imágenes ---
    samples_path = outdir / "samples.png"
    model_for_sampling = None
    if (outdir / "model_ema.pth").exists():
        ema = FMUNet(
            channels=channels, num_residual_layers=depth,
            t_embed_dim=40, y_embed_dim=40,
            in_channels=C, out_channels=C, num_classes=1
        ).to(device)
        ema.load_state_dict(torch.load(outdir / "model_ema.pth", map_location=device))
        model_for_sampling = ema
    else:
        model_for_sampling = unet

    sample_and_save(
        model_for_sampling, samples_path,
        num=36, size=args.size, channels=C, steps=500, device=device
    )
    print("[ok] saved:", samples_path)


if __name__ == "__main__":
    main()
