import argparse, torch
from pathlib import Path
from torchvision.utils import make_grid, save_image
from src.fm import alpha_beta as ab
from src.fm import probability_path as pp
from src.fm import cvectorfield as cvf
from src.fm import sim_utils as sim
from src.model.unet import FMUNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--out", type=str, default="runs/fm_e2s_uncond/samples.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    C = 1 if args.gray else 3

    # Modelo
    unet = FMUNet(
        channels=[32,64,128],
        num_residual_layers=2,
        t_embed_dim=40,
        y_embed_dim=40,
        in_channels=C,      # <- 1 (gris) o 3 (RGB)
        out_channels=C,     # <- genera misma cantidad de canales
        num_classes=1       # <- unconditional
    ).to(device)
    unet.load_state_dict(torch.load(args.ckpt, map_location=device))
    unet.eval()

    # ODE + simulador
    ode = cvf.LearnedVectorFieldODE(unet)  # sin condición efectiva; y=0 default en trainer/simulator
    ode.y = torch.zeros(args.num, dtype=torch.long, device=device)

    simulator = sim.EulerSimulator(ode)

    # Simple (ruido) vía path simple shape
    # Construimos p_simple desde la path API (truco: sampleamos de N(0,I) del shape correcto)
    x0 = torch.randn(args.num, C, args.size, args.size, device=device)

    ts = torch.linspace(0,1,1000, device=device).view(1,-1,1,1,1).expand(args.num,-1,1,1,1)
    y = torch.zeros(args.num, dtype=torch.long, device=device)   # condición cero

    with torch.no_grad():
        x1 = simulator.simulate(x0, ts)

    grid = make_grid(x1, nrow=int(max(1, args.num**0.5)), normalize=True, value_range=(-1,1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, args.out)
    print("[ok] saved samples:", args.out)

if __name__ == "__main__":
    main()
