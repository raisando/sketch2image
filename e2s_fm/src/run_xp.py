# src/train.py
import argparse, torch, re, json, datetime, copy
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from src.datasets.loaders import make_loaders, ImageDatasetSampler, RightHalfImages
from src.fm import alpha_beta as ab, probability_path as pp
from src.fm.trainer import ImageCFMTrainer, ImageCFMTrainerEpoch
from src.model.unet import FMUNet
from src.tools.utils import sample_and_save, eval_loss, eval_on_loader_with_trainer, _peek

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required                  = True)
    ap.add_argument("--size", type=int, default                        = 64)
    ap.add_argument("--epochs", type=int, default                      = 200)
    ap.add_argument("--batch", type=int, default                       = 64)
    ap.add_argument("--lr", type=float, default                        = 1e-4)
    ap.add_argument("--out", type=str, default                         = "runs/fm_e2s_uncond")
    ap.add_argument("--gray", action                                   = "store_true")
    ap.add_argument("--num_workers", type=int, default                 = 8)
    ap.add_argument("--max_updates", type=int, default = 120_000, help = "tope de pasos de optimización; ignora epochs si se usa >0")
    ap.add_argument("--eval_every", type=int, default  = 5_000, help   = "cada cuántos updates correr validación")
    ap.add_argument("--trainer", choices=["steps","epochs"], default   = "steps")
    ap.add_argument("--patience", type=int, default    = 3, help       = "n° de validaciones SIN mejora antes de parar")
    ap.add_argument("--ema_decay", type=float, default                 = 0.9999)
    ap.add_argument("--grad_clip", type=float, default                 = 1.0)
    ap.add_argument("--no_amp", action                                 = "store_true")
    args = ap.parse_args()

    # === Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"[info] device={device}")
    torch.set_float32_matmul_precision("high")

    # === Loaders ===
    train_loader, val_loader, test_loader = make_loaders(
        args.data_root, size=args.size, batch_size=args.batch, to_gray=args.gray,
        num_workers=args.num_workers
    )

    #_peek(train_loader, "train")
    #_peek(val_loader,   "val")
    #_peek(test_loader,  "test")

    p_train = ImageDatasetSampler(train_loader.dataset).to(device)
    path = pp.GaussianConditionalProbabilityPath(
        p_data=p_train,
        alpha=ab.LinearAlpha(),
        beta=ab.SquareRootBeta()
    ).to(device)

    # === RED ===
    C        = 1 if args.gray else 3
    channels = [32, 64, 128]
    depth    = 2
    unet = FMUNet(
        channels            = channels,
        num_residual_layers = depth,
        t_embed_dim         = 40, y_embed_dim=40,
        in_channels         = C, out_channels=C, num_classes=1
    ).to(device)

    # === Trainer con AMP/EMA ===
    trainer_kwargs = dict(
        ema_decay=args.ema_decay,
        use_amp=not args.no_amp,
    )

    # === Entrenamiento ===
    if args.trainer == "steps":
        trainer = ImageCFMTrainer(path, unet, **trainer_kwargs)
        print("[info] start training")

        history = trainer.train(
            device=device,
            lr=args.lr,
            train_loader=train_loader,
            val_loader=val_loader,
            max_updates=args.max_updates,
            eval_every=args.eval_every,
            patience=args.patience,
            save_dir=args.out
        )
    else:
        trainer = ImageCFMTrainerLight(path, unet, **trainer_kwargs)
        print("[info] start training")
        history = trainer.train(
            device=device,
            lr=args.lr,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            patience=args.patience,
            save_dir=args.out
        )


    # === Guardar historia ===
    outdir = Path(args.out)
    (outdir).mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print("[ok] saved:", metrics_path)

    # === Guardar Plot ===
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

    # === Guardar Pesos ===
    torch.save(unet.state_dict(), outdir / "model.pth")
    print("[ok] saved:", outdir / "model.pth")
    if getattr(trainer, "ema_model", None) is not None:
        torch.save(trainer.ema_model.state_dict(), outdir / "model_ema.pth")
        print("[ok] saved:", outdir / "model_ema.pth")

    # === Evaluación FINAL en TEST  ===
    test_loss_full = eval_on_loader_with_trainer(trainer, test_loader, device, use_ema=True)
    print(f"[report] test_loss (EMA, full test set): {test_loss_full:.6f}")


if __name__ == "__main__":
    main()
