"""
Reference:
    https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
"""
from PIL import Image
import numpy as np
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer as RealESRGAN

from .utils import blend_h
from ..path import SUPERRES_LOCAL_MODELS


def load_model(
    model_name: str, 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    tile: int = 384,
    tile_pad: int = 10,
    pre_pad: int = 0,
    use_half: bool = False,
):
    if model_name.endswith(('_x2','_x2+')):
        scale = 2
    elif model_name.endswith(('_x4','_x4+')):
        scale = 4
    else:
        raise (f"{model_name} is not not supported!")
    model_path = SUPERRES_LOCAL_MODELS[model_name]
    model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                        num_block=23, num_grow_ch=32, scale=scale)
    model = RealESRGAN(
        model=model_arch, scale=scale, 
        model_path=model_path, dni_weight=None,
        tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, 
        half=use_half, device=device,
    )
    return model


def run_upsampling(
    model,
    image: Image.Image,
    extend: int = 10,
    outscale: int = 2,
):
    w, h, *_ = image.size

    image = np.array(image)
    image = np.concatenate([image, image[:, :extend, :]], axis=1)
    image = model.enhance(image, outscale = outscale)[0]
    image = blend_h(image, image, extend * outscale)
    image = Image.fromarray(image[:, :w * outscale, :])
    return image


if __name__ == "__main__":

    image_path = "./temp/00003.png"
    image = Image.open(image_path)

    model_name = f"realesgan_x4+"
    model = load_model(model_name)

    upscale = 4
    output = run_upsampling(model, image, outscale=upscale)
    output.save(image_path.replace('.png', f'_x{upscale}.png'))


