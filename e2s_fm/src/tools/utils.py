# src/tools/utils.py
from __future__ import annotations
from pathlib import Path
import copy
from typing import Optional

import torch
from torchvision.utils import make_grid, save_image

from src.fm import cvectorfield as cvf
from src.fm import sim_utils as sim


from torch.utils.data import random_split

def random_split_dataset(dataset, splits=[0.8, 0.2], seed: int = 42):
    """
    Divide un dataset en train/val/test de manera reproducible.

    Args:
        dataset: instancia de torch.utils.data.Dataset
        splits: tupla con proporciones (train, val, test)
        seed: semilla aleatoria para reproducibilidad

    Returns:
        train_set, val_set, test_set
    """
    print("[info] random_split_dataset with splits:", splits)
    assert abs(sum(splits) - 1.0) < 1e-6, "Las proporciones deben sumar 1.0"
    n = len(dataset)
    lengths = [int(p * n) for p in splits]
    lengths[-1] = n - sum(lengths[:-1])  # ajuste final

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, lengths, generator=generator)


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
    cond_y: Optional[torch.Tensor] = None,   # ← ahora se espera embedding CLIP por muestra (opcional)
    grid_nrow: Optional[int] = None,
    batch_size: int = 64,   # por si quieres muestrear en lotes
) -> Path:
    """
    Genera 'num' imágenes integrando el ODE en dirección reverse (t: 1→0).
    Asume entrenamiento en [-1,1] y prior N(0,I) en t=1.

    Parámetros:
      - cond_y: si se provee, debe ser un tensor float de shape [num, D] (e.g. D=512 para CLIP),
           que representa el embedding condicional por muestra. Si es None, sampling incondicional.
    """
    device = device or next(model.parameters()).device
    model.eval()

    # Vector field ODE con el modelo aprendido
    ode = cvf.LearnedVectorFieldODE(model)

    # Condición (unconditional por defecto). Si y existe, lo pasamos por lote.
    if cond_y is not None:
        cond_y = cond_y.to(device, non_blocking=True)
        # validación simple
        assert cond_y.dim() == 2 and cond_y.shape[0] == num, \
            f"Se esperaba y de shape [num, D], recibido {tuple(cond_y.shape)} con num={num}"
        ode.y = cond_y
    else:
        ode.y = None  # incondicional

    simulator = sim.EulerSimulator(ode)

    # Estado inicial: prior en t=1
    ts = torch.linspace(0.0, 1.0, steps, device=device).view(1, -1, 1, 1, 1)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Batching opcional para no pasar 'num' enorme de una:
    imgs = []
    nrow = grid_nrow or int(max(1, num**0.5))
    done = 0
    while done < num:
        b = min(batch_size, num - done)
        x0 = torch.randn(b, channels, size, size, device=device)  # t=1
        ts_b = ts[:, :, :, :, :].expand(b, -1, 1, 1, 1)

        # Condición por batch (si aplica)
        if cond_y is not None:
            ode.y = cond_y[done:done + b]  # [b, D]
        else:
            ode.y = None

        xT = simulator.simulate(x0, ts_b)
        imgs.append(xT)
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


# --- sanity prints: tamaños y primeras rutas ---
def _peek(loader, name):
    ds = loader.dataset
    print(f"[{name}] n={len(ds)} root≈{getattr(ds, 'root', None)}")
    try:
        ex = ds[0]
        print(f"[{name}] ejemplo path={ex.get('path', 'N/A')}, x.shape={ex['x'].shape}, "
            f"x.min={float(ex['x'].min()):.2f}, x.max={float(ex['x'].max()):.2f}")
    except Exception as e:
        print(f"[{name}] peek error:", e)
