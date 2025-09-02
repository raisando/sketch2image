import torch
import torch.nn as nn
from diffusers import UNet2DModel


class CondUNet(nn.Module):
    def __init__(self, in_channels_total=4, out_channels=3, sample_size=256):
        super().__init__()
        self.unet = UNet2DModel(
                sample_size=sample_size,
                in_channels=in_channels_total,
                out_channels=out_channels,
                layers_per_block=2,
                block_out_channels=(128, 256, 256, 256),
                down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )


    def forward(self, x_t, t, sketch):
        # si el sketch tiene 1 canal: concat → [B, 3+1, H, W]
        # si tiene 3 canales: concat → [B, 3+3, H, W] (ajusta in_channels_total)
        x = torch.cat([x_t, sketch], dim=1)
        assert x.shape[1] == self.unet.config.in_channels, f"{x.shape[1]=} vs {self.unet.config.in_channels=}"

        return self.unet(x, t).sample # predicción de ε
