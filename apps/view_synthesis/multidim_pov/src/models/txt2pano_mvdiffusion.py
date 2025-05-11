import os
import sys
import yaml
import argparse
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image

import numpy as np
import torch


#################################
#           Pipeline            #
#################################

from .. import shared
from ..utils import set_seed, clear_torch_cache
from .path import MVDIFF_LOCAL_MODELS, SDIFF_LOCAL_MODELS

# Link to model
current_workdir = Path(__file__).resolve().parent
sys.path.append(str(current_workdir))

from mvdiffusion.src.wrappers import PanoGenerator
from mvdiffusion.tools import generate_pano_video
from mvdiffusion.utils import preprocess_image, rename_attention_keys, multiview_Rs_Ks


def load_config(config_path):
    with open(config_path, 'rb') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    if config['model']['model_id'] == 'stabilityai/stable-diffusion-2-base':
        config['model']['model_id'] = SDIFF_LOCAL_MODELS['sd20_base']
    elif config['model']['model_id'] == 'stabilityai/stable-diffusion-2-inpainting':
        config['model']['model_id'] = SDIFF_LOCAL_MODELS['sd20_inpaint']
    return config


def load_model(model_name: str = "sd2_original/mvdiffusion"):

    # Load config
    config_path = str(current_workdir / 'mvdiffusion/configs/pano_generation.yaml')
    config = load_config(config_path)

    # Load model
    model_path = shared.MVDIFF_LOCAL_MODELS[model_name]
    model_ckpt = torch.load(model_path, map_location='cpu')['state_dict']
    model_ckpt = rename_attention_keys(model_ckpt)
    
    model = PanoGenerator(config)
    model.load_state_dict(model_ckpt, strict=False) # ignore "text_encoder.text_model.embeddings.position_ids"
    model.eval()
    model.to(device=shared.device)
    # model.to(dtype=shared.dtype)

    if not shared.low_vram:
        return model

    # Enable memory efficiency
    print("\n\n Enabling memory efficiency ...")
    model.vae.enable_slicing()
    model.vae.enable_tiling()

    return model


def inference(
    model, 
    prompt: str | list, 
    image_size: int = 256,
    num_views: int = 8,
    diffusion_steps: int = 20, 
    guidance_scale: float = 9.5, 
    **kwargs
):
    images = torch.zeros((1, num_views, image_size, image_size, 3))
    images = images.to(device=shared.device)

    if isinstance(prompt, str):
        prompt = [prompt] * num_views
    else:
        assert isinstance(prompt, (list, tuple))
        assert len(prompt) == num_views

    Rs, Ks = multiview_Rs_Ks(image_size, num_views)
    Ks = torch.tensor(Ks).to(device=shared.device)[None]
    Rs = torch.tensor(Rs).to(device=shared.device)[None]

    if guidance_scale > 0:
        model.guidance_scale = guidance_scale
    
    if diffusion_steps > 0:
        model.diff_timestep = diffusion_steps

    batch_mv = dict(prompt=prompt, images=images, R=Rs, K=Ks)
    generated = model.inference(batch_mv)[0]
    return generated


if __name__ == "__main__":

    prompt = [
        "frontal view of modern conference stage with central LED screen and two side screens, rows of white-covered tables and chairs, overhead spotlights, black floor, black ceiling",
        "front-right angle on sleek stage platform, glowing LED strip edges, abstract visuals on screens, white tables and chairs, blue spotlight beams, black floor, black ceiling",
        "right-side view of stage, prominent side screen, aligned rows of covered tables and chairs, atmospheric blue lighting, overhead light fixtures, black floor, black ceiling",
        "rear-right perspective showing stage side and back, seating area extending, abstract LED patterns, moody blue spotlights, black floor, black ceiling",
        
        "view facing a black wall with two exit doors on either side, cool blue ambient lighting, subtle standees, black floor, black ceiling",
        "rear-left angle capturing seating rows and stage backdrop, left side screen visible, blue ambient glow, overhead beams, black floor, black ceiling",
        "left-side view focusing on side screen, tables and chairs in profile, glowing platform edge, atmospheric blue lighting, black floor, black ceiling",
        "front-left vantage of stage, side screen on left, rows of white tables and chairs, vivid blue spotlights overhead, black floor, black ceiling",
    ]

    resolution = 512
    num_views = 8

    # Load model
    model_name = "sd2_original/mvdiffusion"
    model = load_model(model_name)

    # Run inference
    gen_params = dict(image_size=resolution, num_views=num_views, 
                    diffusion_steps=10, guidance_scale=9.5)
    generated = inference(model, prompt, **gen_params)

    # Save output
    print('\nSaving outputs ...')
    prefix = 'mvdiff_panogen_'
    image_paths = []
    for i in range(num_views):
        image_path = f"./temp/{prefix}view{i+1:02d}.png"
        image_paths.append(image_path)
        im = Image.fromarray(generated[i])
        im.save(image_path)

    generate_pano_video(image_paths, './temp', prefix)


