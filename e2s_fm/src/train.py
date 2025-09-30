# train.py
import argparse, json, datetime
from pathlib import Path
import matplotlib.pyplot as plt
import torch

from src.datasets.right_half_only import make_loaders_mnist, make_loaders, ImageDatasetSampler
from src.fm import alpha_beta as ab, probability_path as pp
from src.fm.trainer import ImageCFMTrainerLight
from src.model.unet import FMUNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default="runs/fm_light_min")
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    print(f"[info] device={device}")

    # Fashion MNIST LOADER
    '''train_loader = make_loaders_mnist(
        data_root=args.data_root,
        size=28, batch_size=64, num_workers=8,
        class_filter=[0]
    )'''

    # Edges2Shoes LOADER
    train_loader,_,_ = make_loaders(
        data_root=args.data_root,
        size=28, batch_size=64, to_gray=args.gray, num_workers=8)

    # p_data y Path
    p_train = ImageDatasetSampler(train_loader.dataset).to(device)
    path_obj = pp.GaussianConditionalProbabilityPath(
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

    # Entrenamiento mínimo
    trainer = ImageCFMTrainerLight(path=path_obj, model=unet)
    history = trainer.train(
        device=device, lr=args.lr, epochs=args.epochs,
        train_loader=train_loader
    )

    # Guardar últimos pesos
    torch.save(unet.state_dict(), outdir / "model.pth")
    print("[ok] saved:", outdir / "model.pth")

    # Guardar métricas y plot (train loss por epoch)
    metrics_path = outdir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(history, f, indent=2)
    print("[ok] saved:", metrics_path)

    plt.figure(figsize=(8,5))
    plt.plot(history["train"], label="train")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Flow Matching - train loss")
    plt.legend(loc="upper right")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    hparams = (f"res={args.size}, C={C}\n"
                f"channels={channels}, depth={depth}\n"
                f"batch={args.batch}, lr={args.lr}\n"
                f"epochs={args.epochs}, device={device}\n"
                f"{stamp}")
    plt.gcf().text(0.98, 0.02, hparams, ha="right", va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="gray"))
    diag_path = outdir / "diagnostics.png"
    plt.tight_layout(); plt.savefig(diag_path, dpi=150); plt.close()
    print("[ok] saved:", diag_path)

    # Nota: el sample lo harás con tu sample.py cargando outdir/"model.pth"

if __name__ == "__main__":
    main()
