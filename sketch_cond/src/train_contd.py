# train_contd.py
import os, re, argparse, json, datetime
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
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        loss = trainer.get_train_loss(batch, device)
        total += float(loss.item())
        n += 1
        if max_batches is not None and (i + 1) >= max_batches:
            break
    if was_training:
        model.train()
    return (total / max(1, n)) if n > 0 else float("nan")

def infer_start_epoch_from_ckpt(ckpt_path: Path) -> int:
    """
    Intenta inferir la epoch desde nombres tipo: ckpt_e0100.pth, e100.pth, ckpt_e100_anything.pth
    Si no puede, retorna None y el caller puede usar --start_epoch.
    """
    m = re.search(r"[eE](\d+)", ckpt_path.stem)
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50, help="Épocas ADICIONALES a entrenar desde el checkpoint")
    ap.add_argument("--batch", type=int, default=64)     # batch POR GPU
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default="runs/fm_light_min")
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8)

    ap.add_argument("--ckpt_every", type=int, default=10, help="Guardar checkpoint cada N épocas (continuará numeración)")
    ap.add_argument("--val_every", type=int, default=10, help="Correr validación cada N épocas")
    ap.add_argument("--val_max_batches", type=int, default=None, help="(opcional) Limitar batches en validación para acelerar")

    # 🔹 nuevo: resume
    ap.add_argument("--resume", type=str, required=True, help="Ruta al checkpoint .pth (guardado previamente)")
    ap.add_argument("--start_epoch", type=int, default=None, help="Epoch de la que parte el checkpoint (si no se infiere del nombre)")
    args = ap.parse_args()

    # --- DDP setup
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    if is_main():
        print(f"[info] device={device}")

    # --- Loaders (idéntico a tu train.py actual)
    train_loader, val_loader = make_loaders_coco_text_distributed(
        data_root=args.data_root,
        size=128,
        batch_size=args.batch,
        num_workers=args.num_workers,
        val_ratio=0.01,
        embeds_pt="data/coco2017_5cls/cache/clip_most_common_class_embeds_",
        distributed=True
    )

    # --- p_data y probability path
    base_ds = train_loader.dataset.dataset if isinstance(train_loader.dataset, Subset) else train_loader.dataset
    p_train = ImageDatasetSampler(base_ds).to(device)
    path_obj = pp.GaussianConditionalProbabilityPath(
        p_data=p_train, alpha=ab.LinearAlpha(), beta=ab.SquareRootBeta()
    ).to(device)

    # --- Modelo
    C = 1 if args.gray else 3
    channels = [32, 64, 128, 256, 512]
    depth = 2
    model = FMUNetCOCO(
        channels=channels,
        num_residual_layers=depth,
        t_embed_dim=40, y_embed_dim=40,
        in_channels=C, out_channels=C,
        clip_dim=512, num_classes=1
    ).to(device)

    # --- Cargar checkpoint en el modelo (antes de DDP está OK también, pero así es explícito)
    ckpt_path = Path(args.resume)
    assert ckpt_path.exists(), f"Checkpoint no existe: {ckpt_path}"
    state = torch.load(ckpt_path, map_location=device)
    # si fue guardado desde DDP: keys sin 'module.'; si tuvieran 'module.' y falla, intentamos limpiar
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # intenta cargar removiendo/el agregando 'module.'
        from collections import OrderedDict
        new_state = OrderedDict()
        if all(k.startswith("module.") for k in state.keys()):
            for k, v in state.items():
                new_state[k[len("module."):]] = v
        else:
            for k, v in state.items():
                new_state["module."+k] = v
        try:
            model.load_state_dict(new_state)
        except Exception:
            raise e

    # --- DDP wrap
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    trainer = ImageCFMTrainerLight(path=path_obj, model=model, use_amp=True)

    # --- Historia: si existe metrics.json en --out, lo seguimos; si no, empezamos nuevo
    outdir = Path(args.out)
    if is_main():
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    history = {"train": [], "val": []}
    metrics_path = outdir / "metrics.json"
    if is_main() and metrics_path.exists():
        try:
            prev = json.load(open(metrics_path))
            # muy tolerante con formatos
            if isinstance(prev, dict):
                history["train"] = list(prev.get("train", []))
                history["val"]   = list(prev.get("val", []))
        except Exception as _:
            pass

    # --- Epoch inicial: inferir del nombre o usar flag
    inferred = infer_start_epoch_from_ckpt(ckpt_path)
    start_epoch = args.start_epoch if args.start_epoch is not None else (inferred if inferred is not None else 0)
    if is_main():
        print(f"[info] reanudando desde epoch={start_epoch} (inferido de {ckpt_path.name})")

    # --- Entrenamiento por bloques (igual que tu train.py)
    add_epochs  = max(0, int(args.epochs))
    ckpt_every  = max(1, int(args.ckpt_every))
    val_every   = max(1, int(args.val_every))

    done = 0
    global_epoch = start_epoch  # 🔸 continúa numeración

    try:
        while done < add_epochs:
            block = min(ckpt_every, add_epochs - done)

            if hasattr(getattr(train_loader, "sampler", None), "set_epoch"):
                getattr(train_loader, "sampler").set_epoch(global_epoch)

            block_history = trainer.train(
                device=device,
                lr=args.lr,
                epochs=block,
                train_loader=train_loader,
                use_amp=False,  # igual que tu script
            )
            if isinstance(block_history, dict) and "train" in block_history:
                history["train"].extend(block_history["train"])
            elif isinstance(block_history, list):
                history["train"].extend(block_history)
            else:
                history["train"].extend([float("nan")] * block)

            done += block
            global_epoch += block  # 🔸 avanza numeración global

            # validación
            if (global_epoch % val_every == 0) or (done == add_epochs):
                val_loss = evaluate(trainer, model, val_loader, device, max_batches=args.val_max_batches)
                if dist.is_initialized():
                    t = torch.tensor([val_loss], dtype=torch.float32, device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.AVG)
                    val_loss = t.item()
                if is_main():
                    history["val"].append({"epoch": global_epoch, "loss": val_loss})
                    print(f"[val] epoch={global_epoch} loss={val_loss:.6f}")

            # checkpoint
            if is_main() and (global_epoch % ckpt_every == 0 or done == add_epochs):
                ckpt_out = outdir / f"checkpoints/ckpt_e{global_epoch:04d}.pth"
                torch.save(model.module.state_dict(), ckpt_out)
                print("[ok] checkpoint saved:", ckpt_out)

        # --- guardar finales
        if is_main():
            # modelo final
            torch.save(model.module.state_dict(), outdir / "model.pth")
            print("[ok] saved:", outdir / "model.pth")

            # métricas (append/overwrite)
            with metrics_path.open("w") as f:
                json.dump(history, f, indent=2)
            print("[ok] saved:", metrics_path)

            # plot
            plt.figure(figsize=(8,5))
            plt.plot(history["train"], label="train")
            if len(history["val"]) > 0:
                xs = [v["epoch"] for v in history["val"]]
                ys = [v["loss"] for v in history["val"]]
                plt.plot(xs, ys, marker="o", linestyle="--", label="val")
            plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Flow Matching - loss (continued)")
            plt.legend(loc="upper right")

            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            hparams = (f"res={args.size}, C={C}\n"
                       f"channels={channels}, depth={depth}\n"
                       f"batch/GPU={args.batch}, lr={args.lr}\n"
                       f"added_epochs={add_epochs}, start_epoch={start_epoch}, device={device}\n"
                       f"{stamp}")
            plt.gcf().text(0.98, 0.02, hparams, ha="right", va="bottom",
                        bbox=dict(boxstyle="round", fc="white", ec="gray"))
            diag_path = outdir / "diagnostics.png"
            plt.tight_layout(); plt.savefig(diag_path, dpi=150); plt.close()
            print("[ok] saved:", diag_path)

    finally:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
