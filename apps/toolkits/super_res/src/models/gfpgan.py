"""
Reference:
    https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py
"""
from PIL import Image

import numpy as np
import torch

from pathlib import Path
xtra_dir = str(Path(__file__).resolve().parents[5] / 'extra')

import sys
sys.path.append(xtra_dir)

from gfpgan import GFPGANer as GFPGAN

from .utils import blend_h
from ..path import SUPERRES_LOCAL_MODELS


def load_model(
    model_name: str, 
    upscale: int = 2,
    upsampler = None,   # RealESRGAN for background upsampling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    assert model_name.startswith(('gfpgan_', 'restoreformer'))

    model_path = SUPERRES_LOCAL_MODELS[model_name]
    model_arch = 'clean' if model_name.startswith('gfpgan_') else 'RestoreFormer'

    model = GFPGAN(
        arch=model_arch, model_path=model_path, 
        upscale=upscale, bg_upsampler=upsampler,
        channel_multiplier=2,
    )
    return model


def run_upsampling(
    model,
    image: Image.Image,
    adj_weight: float = 0.5,
    is_aligned: bool = False,
    only_center_face: bool = False,
):
    cropped_faces, \
    restored_faces, \
    restored_image = model.enhance(
        image, has_aligned=is_aligned,
        only_center_face=only_center_face,
        paste_back=True, weight=adj_weight,
    )
    return restored_image


if __name__ == "__main__":

    image_path = "./temp/barca.jpg"
    image = Image.open(image_path)
    image = np.asarray(image)

    upscale = 4
    model_name = f"gfpgan_v13"
    model = load_model(model_name, upscale=upscale)

    output = run_upsampling(model, image)
    output = Image.fromarray(output)
    output.save(image_path.replace('.jpg', f'_x{upscale}.jpg'))


