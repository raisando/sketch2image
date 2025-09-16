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
    batch_size: int = 64,   # por si quieres muestrear en lotes
) -> Path:
    """
    Genera 'num' imágenes integrando el ODE en dirección reverse (t: 1→0).
    Asume entrenamiento en [-1,1] y prior N(0,I) en t=1.
    """
    device = device or next(model.parameters()).device
    model.eval()

    # Vector field ODE con el modelo aprendido
    ode = cvf.LearnedVectorFieldODE(model)

    # Condición (unconditional por defecto)
    if y is None:
        y = torch.zeros(num, dtype=torch.long, device=device)
    ode.y = y

    simulator = sim.EulerSimulator(ode)

    # Estado inicial: prior en t=1
    # OJO: el simulador asume que el primer tiempo en ts corresponde al estado inicial
    # así que vamos a pasar ts decreciente (1→0) y x_init ~ N(0, I)
    ts = torch.linspace(1.0, 0.0, steps, device=device).view(1, -1, 1, 1, 1)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Batching opcional para no pasar 'num' enorme de una:
    imgs = []
    nrow = grid_nrow or int(max(1, num**0.5))
    done = 0
    while done < num:
        b = min(batch_size, num - done)
        xT = torch.randn(b, channels, size, size, device=device)  # t=1
        ts_b = ts[:, :, :, :, :].expand(b, -1, 1, 1, 1)
        x0 = simulator.simulate(xT, ts_b)  # integra 1→0
        imgs.append(x0)
        done += b

    x_all = torch.cat(imgs, dim=0)[:num]

    grid = make_grid(x_all, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, out_png)
    return out_png



# src/tools/utils.py
import torch

@torch.no_grad()
def eval_loader_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        # Si tu modelo expone una API de pérdida directa:
        # loss = model.loss(x)  # adapta si existe
        # Como tu loss vive en el trainer, replicamos la lógica de _loss_from_batch:
        B = x.size(0)
        eps = 1e-3
        t_img = torch.rand(B, 1, 1, 1, device=device).clamp(eps, 1.0 - eps)
        # Necesitamos el **path** para generar x_t y u_ref; aquí hay dos opciones:
        # (A) Pasar el 'trainer' y reutilizar su path, o
        # (B) Crear una pequeña función eval que reciba 'path'.
        # Para mínima invasión, defínela en el mismo archivo o pásala por fuera.
        raise NotImplementedError("Ver nota en la respuesta: usa la variante con 'trainer' más abajo.")


# src/tools/utils.py
@torch.no_grad()
def eval_on_loader_with_trainer(trainer, loader, device, use_ema=True):
    """
    Evalúa pérdida promedio recorriendo TODO el loader con el 'trainer' dado.
    Usa ema_model si está disponible.
    """
    model = trainer.ema_model if (use_ema and getattr(trainer, "ema_model", None) is not None) else trainer.model
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        loss = trainer._loss_from_batch(x, device)
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(1, n)
