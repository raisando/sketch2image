# src/tools/utils.py
from __future__ import annotations
from pathlib import Path
import copy
from typing import Optional

import torch
from torchvision.utils import make_grid, save_image

from src.fm import cvectorfield as cvf
from src.fm import sim_utils as sim


@torch.no_grad()
def eval_loss(
    trainer,
    p_data,
    device: torch.device,
    *,
    batches: int = 50,
    batch_size: int = 64,
    use_ema: bool = True,
) -> float:
    """
    Evalúa la loss promedio sobre 'batches' usando el path del trainer
    pero con p_data reemplazado por el p_data proporcionado (val/test).

    Retorna la loss promedio (float).
    """
    # Elegir modelo (EMA si existe y se pide)
    model = trainer.ema_model if (use_ema and getattr(trainer, "ema_model", None) is not None) else trainer.model
    model.eval()

    # Guardar/restaurar path del trainer (clon superficial para cambiar p_data)
    old_path = trainer.path
    path_eval = copy.copy(trainer.path)
    path_eval.p_data = p_data.to(device)
    trainer.path = path_eval

    acc = 0.0
    for _ in range(max(1, batches)):
        acc += trainer._one_batch_loss(batch_size, device).item()

    # Restaurar
    trainer.path = old_path
    return acc / max(1, batches)


@torch.no_grad()
def sample_and_save(
    model: torch.nn.Module,
    out_png: Path,
    *,
    num: int = 36,
    size: int = 64,
    channels: int = 3,
    steps: int = 500,
    device: Optional[torch.device] = None,
    y: Optional[torch.Tensor] = None,
    grid_nrow: Optional[int] = None,
) -> Path:
    """
    Genera 'num' imágenes vía ODE Euler y guarda un grid en 'out_png'.
    - 'y' es opcional (condiciones). Si None -> ceros (unconditional).
    - Usa normalize=True y value_range=(-1,1) asumiendo training en [-1,1].
    """
    device = device or next(model.parameters()).device
    model.eval()

    # Vector field ODE con el modelo aprendido
    ode = cvf.LearnedVectorFieldODE(model)
    # Para uncond, y = 0; si pasas y (por clase), úsala.
    if y is None:
        y = torch.zeros(num, dtype=torch.long, device=device)
    ode.y = y

    simulator = sim.EulerSimulator(ode)

    # Estado inicial: ruido gaussiano N(0,I)
    x0 = torch.randn(num, channels, size, size, device=device)
    ts = torch.linspace(0, 1, steps, device=device).view(1, -1, 1, 1, 1).expand(num, -1, 1, 1, 1)

    x1 = simulator.simulate(x0, ts)

    # Guardar grid
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    nrow = grid_nrow or int(max(1, num**0.5))
    grid = make_grid(x1, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, out_png)
    return out_png
