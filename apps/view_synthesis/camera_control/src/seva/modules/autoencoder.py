import torch
from torch import nn
from diffusers.models import AutoencoderKL  # type: ignore

from ... import shared


class AutoEncoder(nn.Module):

    scale_factor: float = 0.18215
    downsample: int = 8

    def __init__(
        self, 
        model_path: str , 
        chunk_size: int | None = None,
        **vae_kwargs
    ):
        super().__init__()
        self.module = AutoencoderKL.from_pretrained(model_path, subfolder="vae", **vae_kwargs)
        self.module.eval().requires_grad_(False)  # type: ignore
        if shared.low_vram:
            self.module.enable_slicing()
            self.module.enable_tiling()

        self.chunk_size = chunk_size

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.module.encode(x).latent_dist.mean * self.scale_factor

    def encode(self, x: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        chunk_size = chunk_size or self.chunk_size
        if chunk_size is not None:
            return torch.cat([self._encode(x_chunk) 
                                        for x_chunk in x.split(chunk_size)], dim=0)
        else:
            return self._encode(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.module.decode(z / self.scale_factor).sample  # type: ignore

    def decode(self, z: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        chunk_size = chunk_size or self.chunk_size
        if chunk_size is not None:
            return torch.cat([self._decode(z_chunk) 
                                        for z_chunk in z.split(chunk_size)], dim=0)
        else:
            return self._decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

