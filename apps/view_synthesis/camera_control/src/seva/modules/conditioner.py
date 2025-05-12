import os.path as osp

import kornia
from open_clip import create_model_and_transforms

import torch
from torch import nn

from ...path import LOCAL_IMAGE_ENCODERS


class CLIPConditioner(nn.Module):

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self, 
        version: str = "clip-vit-h14-laion2B", 
        model_name: str = "ViT-H-14",
    ):
        super().__init__()

        # Check local checkpoint
        local_path = LOCAL_IMAGE_ENCODERS[version]
        if osp.isfile(local_path):
            version = local_path
        else:
            version = "laion2b_s32b_b79k"
            # version = 'hf-hub:' + REMOTE_IMAGE_ENCODERS[version].replace('https://huggingface.co/', '')        

        self.module = create_model_and_transforms(model_name, pretrained=version)[0]
        self.module.eval().requires_grad_(False)  # type: ignore

        self.register_buffer("mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer("std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = kornia.geometry.resize(x, (224, 224), interpolation="bicubic", align_corners=True, antialias=True)
        x = kornia.enhance.normalize((x + 1.0) / 2.0, self.mean, self.std)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.module.encode_image(x)
        return x

