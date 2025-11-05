# train.py
import os, argparse, json, datetime
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset

from src.datasets.loaders import make_loaders_coco_text_distributed, ImageDatasetSampler
from src.fm import alpha_beta as ab, probability_path as pp
from src.fm.trainer import ImageCFMTrainerLight
from src.model.unet import FMUNetCOCO


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

@torch.no_grad()
def evaluate(trainer, model, val_loader, device, max_batches=None):
    """Valida promediando la misma loss que se usa en train."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        loss = trainer.get_train_loss(batch, device)  # reutilizamos la misma métrica/loss
        total += float(loss.item())
        n += 1
        if max_batches is not None and (i + 1) >= max_batches:
            break
    if was_training:
        model.train()
    # Evitar NaNs si hubiese 0 batches
    return (total / max(1, n)) if n > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)     # batch POR GPU
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default="runs/fm_light_min")
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8)

    ap.add_argument("--ckpt_every", type=int, default=10, help="Guardar checkpoint cada N épocas")
    ap.add_argument("--val_every", type=int, default=10, help="Correr validación cada N épocas")
    ap.add_argument("--val_max_batches", type=int, default=None, help="(opcional) Limitar batches en validación para acelerar")
    args = ap.parse_args()

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    if is_main():
        print(f"[info] device={device}")

    # --- COCO17 LOADER (distribuido)
    train_loader, val_loader = make_loaders_coco_text_distributed(
        data_root=args.data_root,
        size=128,
        batch_size=args.batch,             # batch por GPU
        num_workers=args.num_workers,
        val_ratio=0.01,
        embeds_pt="data/coco2017_5cls/cache/clip_most_present_class_index_embeds_",
        distributed=True
    )

    # --- p_data y probability path
    base_ds = train_loader.dataset.dataset if isinstance(train_loader.dataset, Subset) else train_loader.dataset
    p_train = ImageDatasetSampler(base_ds).to(device)
    path_obj = pp.GaussianConditionalProbabilityPath(
        p_data=p_train, alpha=ab.LinearAlpha(), beta=ab.SquareRootBeta()
    ).to(device)

    C = 1 if args.gray else 3
    channels = [32, 64, 128, 256, 512]
    depth = 2
    unet = FMUNetCOCO(
        channels=channels,
        num_residual_layers=depth,
        t_embed_dim=40, y_embed_dim=40,
        in_channels=C, out_channels=C,
        clip_dim=512, num_classes=1
    ).to(device)

    #unet = torch.compile(unet,mode="reduce-overhead", fullgraph=False)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    trainer = ImageCFMTrainerLight(path=path_obj, model=unet, use_amp=True)

    # Historial extendido con validación
    history = {"train": [], "val": []}

    # --- Entrenamiento en BLOQUES para poder validar y hacer checkpoint ---
    total_epochs = args.epochs
    ckpt_every = max(1, int(args.ckpt_every))
    val_every = max(1, int(args.val_every))

    # Carpeta de salida (solo rank 0 la crea)
    outdir = Path(args.out)
    if is_main():
        outdir.mkdir(parents=True, exist_ok=True)

    # Bucle por bloques
    done = 0
    global_epoch = 0
    while done < total_epochs:
        # Tamaño del bloque: entrenamos como mucho ckpt_every, pero sin pasarnos del total requerido
        block = min(ckpt_every, total_epochs - done)

        # Si el sampler es distribuido, setear el epoch puede ayudar a shuffling determinístico
        if hasattr(getattr(train_loader, "sampler", None), "set_epoch"):
            getattr(train_loader, "sampler").set_epoch(global_epoch)

        # Entrenar 'block' épocas más
        block_history = trainer.train(
            device=device,
            lr=args.lr,
            epochs=block,
            train_loader=train_loader,
            use_amp=False,
        )
        # Acumular historial de entrenamiento (asumo devuelve {"train": [...]})
        if isinstance(block_history, dict) and "train" in block_history:
            history["train"].extend(block_history["train"])
        elif isinstance(block_history, list):  # fallback por si devuelve lista
            history["train"].extend(block_history)
        else:
            # Si por alguna razón no devuelve nada, protegemos
            history["train"].extend([float("nan")] * block)

        done += block
        global_epoch += block

        # VALIDACIÓN si corresponde
        if is_main() and ((global_epoch % val_every == 0) or (done == total_epochs)):
            val_loss = evaluate(
                trainer=trainer,
                model=unet,        # still wrapped in DDP, but that's okay
                val_loader=val_loader,
                device=device,
                max_batches=args.val_max_batches
            )
            history["val"].append({"epoch": global_epoch, "loss": val_loss})
            print(f"[val] epoch={global_epoch} loss={val_loss:.6f}")

            # Registrar solo una vez
            if is_main():
                history["val"].append({"epoch": global_epoch, "loss": val_loss})
                print(f"[val] epoch={global_epoch} loss={val_loss:.6f}")

        # CHECKPOINT si corresponde
        if is_main() and (global_epoch % ckpt_every == 0 or done == total_epochs):
            outdir_ckp = outdir / f"checkpoints"
            outdir_ckp.mkdir(parents=True, exist_ok=True)
            ckpt_path = outdir / f"checkpoints/ckpt_e{global_epoch:04d}.pth"
            torch.save(unet.module.state_dict(), ckpt_path)
            print("[ok] checkpoint saved:", ckpt_path)

    # --- Guardar resultados finales solo en rank 0 ---
    if is_main():
        # último modelo "model.pth"
        torch.save(unet.module.state_dict(), outdir / "model.pth")
        print("[ok] saved:", outdir / "model.pth")

        metrics_path = outdir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(history, f, indent=2)
        print("[ok] saved:", metrics_path)

        # Curva train y val
        plt.figure(figsize=(8,5))
        plt.plot(history["train"], label="train")
        if len(history["val"]) > 0:
            xs = [v["epoch"] for v in history["val"]]
            ys = [v["loss"] for v in history["val"]]
            plt.plot(xs, ys, marker="o", linestyle="--", label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Flow Matching - loss")
        plt.legend(loc="upper right")

        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        hparams = (f"res={args.size}, C={C}\n"
                   f"channels={channels}, depth={depth}\n"
                   f"batch/GPU={args.batch}, lr={args.lr}\n"
                   f"epochs={args.epochs}, device={device}\n"
                   f"{stamp}")
        plt.gcf().text(0.98, 0.02, hparams, ha="left", va="top",
                       bbox=dict(boxstyle="round", fc="white", ec="gray"))
        diag_path = outdir / "diagnostics.png"
        plt.tight_layout(); plt.savefig(diag_path, dpi=150); plt.close()
        print("[ok] saved:", diag_path)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
