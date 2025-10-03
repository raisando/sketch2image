# sanity_cfm.py
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.datasets.loaders import RightHalfImages, ImageDatasetSampler
from src.fm import alpha_beta as ab, probability_path as pp
from src.model.unet import FMUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) dataset y un batch real
data_root = "/home/raisando/tesis/e2s_fm/data"
ds = RightHalfImages(Path(data_root) / "train", size=64, to_gray=False)
assert len(ds) > 0, "Dataset vacío; revisa rutas/extensiones."
dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
batch = next(iter(dl))
z = batch["x"].to(device)  # [B,C,H,W] en [-1,1]

# 2) sampler para path
p_data = ImageDatasetSampler(ds).to(device)

# 3) path y modelo (mismos hparams que en train.py)
path = pp.GaussianConditionalProbabilityPath(
    p_data=p_data,
    alpha=ab.LinearAlpha(),
    beta=ab.SquareRootBeta()
).to(device)

unet = FMUNet(
    channels=[32, 64, 128],
    num_residual_layers=2,
    t_embed_dim=40, y_embed_dim=40,
    in_channels=3, out_channels=3, num_classes=1
).to(device)

ckpt = "/home/raisando/tesis/e2s_fm/runs/fm_e2s_uncond_full/model_ema.pth"
state = torch.load(ckpt, map_location=device, weights_only=True)
unet.load_state_dict(state)
unet.eval()

@torch.no_grad()
def mse(a, b):
    return torch.mean((a - b) ** 2).item()

with torch.no_grad():
    for tval in [0.1, 0.3, 0.5, 0.7, 0.9]:
        t = torch.full((z.size(0), 1, 1, 1), tval, device=device)
        x_t   = path.sample_conditional_path(z, t)            # [B,C,H,W]
        u_ref = path.conditional_vector_field(x_t, z, t)      # [B,C,H,W]
        y     = torch.zeros(z.size(0), dtype=torch.long, device=device)
        u_pred= unet(x_t, t, y)
        print(f"t={tval:.1f}  MSE(u_pred, u_ref) = {mse(u_pred, u_ref):.6f}")
