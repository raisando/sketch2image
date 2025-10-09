import torch
import torch.nn as nn
import math
from typing import Optional, List, Type, Tuple, Dict

class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1) # (bs, 1)
        freqs = t * self.weights * 2 * math.pi # (bs, half_dim)
        sin_embed = torch.sin(freqs) # (bs, half_dim)
        cos_embed = torch.cos(freqs) # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # (bs, dim)

class ResidualLayer(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        # Converts (bs, time_embed_dim) -> (bs, channels)
        self.time_adapter = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, channels)
        )
        # Converts (bs, y_embed_dim) -> (bs, channels)
        self.y_adapter = nn.Sequential(
            nn.Linear(y_embed_dim, y_embed_dim),
            nn.SiLU(),
            nn.Linear(y_embed_dim, channels)
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        res = x.clone() # (bs, c, h, w)

        # Initial conv block
        x = self.block1(x) # (bs, c, h, w)

        # Add time embedding
        t_embed = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1) # (bs, c, 1, 1)
        x = x + t_embed

        # Add y embedding (conditional embedding)
        y_embed = self.y_adapter(y_embed).unsqueeze(-1).unsqueeze(-1) # (bs, c, 1, 1)
        x = x + y_embed

        # Second conv block
        x = self.block2(x) # (bs, c, h, w)

        # Add back residual
        x = x + res # (bs, c, h, w)

        return x

class Encoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_in, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])
        self.downsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c_in, h, w) -> (bs, c_in, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        # Downsample: (bs, c_in, h, w) -> (bs, c_out, h // 2, w // 2)
        x = self.downsample(x)

        return x

class Midcoder(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c, h, w) -> (bs, c, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x

class Decoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1))
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_out, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Upsample: (bs, c_in, h, w) -> (bs, c_out, 2 * h, 2 * w)
        x = self.upsample(x)

        # Pass through residual blocks: (bs, c_out, h, w) -> (bs, c_out, 2 * h, 2 * w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x
'''
class MNISTUNet(util.ConditionalVectorField):
    def __init__(self, channels: List[int], num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        # Initial convolution: (bs, 1, 32, 32) -> (bs, c_0, 32, 32)
        self.init_conv = nn.Sequential(nn.Conv2d(1, channels[0], kernel_size=3, padding=1), nn.BatchNorm2d(channels[0]), nn.SiLU())

        # Initialize time embedder
        self.time_embedder = FourierEncoder(t_embed_dim)

        # Initialize y embedder
        self.y_embedder = nn.Embedding(num_embeddings = 11, embedding_dim = y_embed_dim)

        # Encoders, Midcoders, and Decoders
        encoders = []
        decoders = []
        for (curr_c, next_c) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, y_embed_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)

        # Final convolution
        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, 1, 32, 32)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, 1, 32, 32)
        """
        # Embed t and y
        t_embed = self.time_embedder(t) # (bs, time_embed_dim)
        y_embed = self.y_embedder(y) # (bs, y_embed_dim)

        # Initial convolution
        x = self.init_conv(x) # (bs, c_0, 32, 32)

        residuals = []

        # Encoders
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed) # (bs, c_i, h, w) -> (bs, c_{i+1}, h // 2, w //2)
            residuals.append(x.clone())

        # Midcoder
        x = self.midcoder(x, t_embed, y_embed)

        # Decoders
        for decoder in self.decoders:
            res = residuals.pop() # (bs, c_i, h, w)
            x = x + res
            x = decoder(x, t_embed, y_embed) # (bs, c_i, h, w) -> (bs, c_{i-1}, 2 * h, 2 * w)

        # Final convolution
        x = self.final_conv(x) # (bs, 1, 32, 32)

        return x
'''

# ====== General ======
class FMUNet(nn.Module):
    def __init__(
        self,
        channels: List[int],
        num_residual_layers: int,
        t_embed_dim: int,
        y_embed_dim: int,
        in_channels: int = 1,
        out_channels: int = 1,
        num_classes: int = 1
    ):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU()
        )
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.y_embedder = nn.Embedding(num_embeddings=num_classes, embedding_dim=y_embed_dim)

        # Encoders / Decoders
        encoders = []
        decoders = []
        for (curr_c, next_c) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, y_embed_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)

        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        x: (bs, C_in, H, W)
        t: (bs, 1, 1, 1)
        y: (bs,)  (clase); uncond -> zeros
        """
        t_embed = self.time_embedder(t)   # (bs, t_embed_dim)
        y_embed = self.y_embedder(y)      # (bs, y_embed_dim)

        x = self.init_conv(x)  # (bs, c_0, H, W)

        residuals = []
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed)
            residuals.append(x.clone())

        x = self.midcoder(x, t_embed, y_embed)

        for decoder in self.decoders:
            res = residuals.pop()
            x = x + res
            x = decoder(x, t_embed, y_embed)

        x = self.final_conv(x)  # (bs, C_out, H, W)
        return x

class FMUNetCOCO(nn.Module):
    """
    UNet para Flow Matching condicionado por texto (embeddings CLIP).
    Interfaz: forward(x, t, y)
      - x: (B, C_in, H, W) en [-1,1]
      - t: (B, 1, 1, 1)
      - y:
          * FloatTensor (B, clip_dim) -> proyección lineal a y_embed_dim
          * LongTensor (B,) -> opcional: IDs de clase (si quisieras compat)
          * None -> unconditional (vector de ceros)

    Args:
      channels           : lista de canales por nivel, p.ej. [64,128,256,512,512]
      num_residual_layers: # de bloques residuales por nivel (no cambia resolución)
      t_embed_dim        : dim del embedding de tiempo (Fourier)
      y_embed_dim        : dim del embedding de condición usado en los bloques
      in_channels/out_channels: canales de imagen
      clip_dim           : dimensión del embedding de CLIP (ViT-B/32=512)
      num_classes        : opcional, si quieres mantener soporte a IDs de clase
    """
    def __init__(
        self,
        channels: List[int],
        num_residual_layers: int,
        t_embed_dim: int,
        y_embed_dim: int,
        in_channels: int = 3,
        out_channels: int = 3,
        clip_dim: int = 512,
        num_classes: int = 1,
    ):
        super().__init__()

        # --- Embeddings ---
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.clip_dim = clip_dim
        self.y_proj   = nn.Linear(self.clip_dim, y_embed_dim)                # CLIP -> y_embed

        # --- Backbone UNet ---
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU()
        )

        encoders, decoders = [], []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, y_embed_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder   = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

    def _make_y_embed(self, y: torch.Tensor | None, B: int, device: torch.device) -> torch.Tensor:
        """
        Devuelve y_embed: [B, y_embed_dim]
          - y Float [B, clip_dim] o [clip_dim]  -> y_proj
          - y Long  [B] (IDs)                    -> y_embedder_ids
          - y None                                -> ceros (uncond)
        """
        # CLIP embeddings
        if y.dtype in (torch.float16, torch.float32, torch.float64):
            y = y.to(device)
            if y.ndim == 1:            # [clip_dim] -> [1, clip_dim] -> expand
                y = y.unsqueeze(0).expand(B, -1)
            elif y.ndim == 2 and y.size(0) == 1:
                y = y.expand(B, -1)
            elif y.ndim != 2 or y.size(0) != B:
                raise ValueError(f"[FMUNetCOCO] y shape incompatible: {tuple(y.shape)} con batch={B}")
            return self.y_proj(y)

        raise TypeError(f"[FMUNetCOCO] dtype de y no soportado: {y.dtype}")

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None):
        """
        x: (B, C_in, H, W)    t: (B,1,1,1)    y: ver _make_y_embed
        """
        B, device = x.size(0), x.device

        t_embed = self.time_embedder(t)
        if y is not None:
            y = y.to(device=device, dtype=torch.long)  # [B, t_embed_dim]
        y_embed = self._make_y_embed(y, B, device)      # [B, y_embed_dim]


        x = self.init_conv(x)                           # [B, C0, H, W]

        residuals = []
        for enc in self.encoders:
            x = enc(x, t_embed, y_embed)
            residuals.append(x.clone())

        x = self.midcoder(x, t_embed, y_embed)

        for dec in self.decoders:
            res = residuals.pop()
            x = x + res
            x = dec(x, t_embed, y_embed)

        x = self.final_conv(x)
        return x
