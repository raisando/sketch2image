# src/fm/cvectorfield.py
from __future__ import annotations
import torch
from torch import nn
from . import probability_path as pp
from . import sim_utils as sim
from . import alpha_beta as ab

class ConditionalVectorFieldODE(sim.ODE):
    """
    ODE con el *vector field de referencia* del path (u_t(x|z)).
    Funciona tanto en vectores [bs, dim] como en imágenes [bs, C, H, W].
    """
    def __init__(self, path: pp.ConditionalProbabilityPath, z: torch.Tensor):
        super().__init__()
        self.path = path
        self.z = z  # (1, ...) o (bs, ...)

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        z = self.z
        if z.shape[0] == 1:
            z = z.expand(bs, *z.shape[1:])
        # path.conditional_vector_field debe manejar shapes de x/t
        return self.path.conditional_vector_field(x, z, t)

class LearnedVectorFieldODE(sim.ODE):
    """
    ODE cuyo drift lo da un *modelo aprendido* (p.ej., FMUNet).
    Admite:
      - forward(x, t)          (modelos MLP planos)
      - forward(x, t, y)       (UNet condicional; en uncond, y=0)
    `simulate(..., y=...)` del simulador se propaga como kwarg.
    """
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        # opcional: self.y = None

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # intenta usar self.y si fue seteado desde fuera
        y = getattr(self, "y", None)
        if y is None:
            y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        try:
            return self.net(x, t, y)   # UNet condicional
        except TypeError:
            return self.net(x, t)

class LangevinFlowSDE(sim.SDE):
    """
    Variante SDE (opcional). Se deja tipado genérico y sin simple_model.
    flow_model: nn.Module con forward(x, t) -> u_t(x)
    """
    def __init__(self, flow_model: nn.Module, sigma: float, alpha: ab.Alpha, beta: ab.Beta):
        super().__init__()
        self.flow_model = flow_model
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        flow_value = self.flow_model(x, t)
        score_value = (self.alpha(t) * flow_value - self.alpha.dt(t) * x)
        denom = (torch.square(self.beta(t)) * self.alpha.dt(t) - self.alpha(t) * self.beta.dt(t) * self.beta(t))
        score_value = score_value / denom
        return flow_value + 0.5 * (self.sigma ** 2) * score_value

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.sigma * torch.randn_like(x)
