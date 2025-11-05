import torch
from pathlib import Path
from src.model.unet import FMUNetCOCO
from src.fm import probability_path as pp, alpha_beta as ab
from src.datasets.loaders import make_loaders_coco_text,make_loaders_coco_text_distributed, ImageDatasetSampler
from src.fm.trainer import loss_fn  # usa tu mismo cálculo de pérdida
from src.fm.distrib_utils import ImageDatasetSampler

@torch.no_grad()
def main():
    ckpt = "runs/fm_coco_s128_b16_e100/model.pth"
    data_root = "data/coco2017"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- loader (idéntico a train.py) ---
    train_loader, val_loader = make_loaders_coco_text_distributed(
        data_root=data_root,
        size=128,
        batch_size=16,
        num_workers=4,
        val_ratio=0.01,
        embeds_pt="data/coco2017/cache/clip_most_common_class_embeds_train.pt",
        distributed=False
    )
    base_ds = val_loader.dataset.dataset
    p_val = ImageDatasetSampler(base_ds).to(device)

    # --- modelo ---
    model = FMUNetCOCO(
        channels=[32, 64, 128, 256, 512],
        num_residual_layers=2,
        t_embed_dim=40, y_embed_dim=40,
        in_channels=3, out_channels=3,
        clip_dim=512
    ).to(device)

    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[ok] cargado checkpoint {ckpt}")

    # --- probability path (igual que training) ---
    path = pp.GaussianConditionalProbabilityPath(
        p_data=p_val,
        alpha=ab.LinearAlpha(),
        beta=ab.SquareRootBeta()
    ).to(device)

    # --- validación ---
    total_loss, n = 0, 0
    for batch in val_loader:
        x0, _, y = batch  # depende de tu Dataset (usa las mismas posiciones que train)
        x0 = x0.to(device)
        y = y.to(device) if y is not None else None

        t = torch.rand(x0.size(0), 1, 1, 1, device=device)
        z = path.sample_conditioning_variable(x0.size(0)).to(device)
        xt = path.sample_conditional_path(z, t)

        u_ref = path.conditional_vector_field(xt, z, t)
        u_pred = model(xt, t, y)

        loss = loss_fn(u_pred, u_ref)
        total_loss += loss.item() * x0.size(0)
        n += x0.size(0)

    print(f"[val] promedio de loss: {total_loss / n:.6f}")

if __name__ == "__main__":
    main()
