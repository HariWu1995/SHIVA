import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from pathlib import Path
current_workdir = Path(__file__).resolve().parent

import json
import yaml
from typing import List, Tuple, Union, Optional

import argparse
from datetime import datetime

import cv2
from PIL import Image

import numpy as np
import torch
torch.manual_seed(0)

from diffusers.hooks import apply_group_offloading
from diffusers.utils.import_utils import is_xformers_available

from .src.wrappers import PanoGenerator, PanoOutpaintor
from .tools.pano_video_generation import generate_video

from .utils import preprocess_image, rename_attention_keys, multiview_Rs_Ks
from ..path import MVDIFF_LOCAL_MODELS, SDIFF_LOCAL_MODELS
from ...utils import clear_torch_cache
from ... import shared


def load_config(config_path):
    with open(config_path, 'rb') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    if config['model']['model_id'] == 'stabilityai/stable-diffusion-2-base':
        config['model']['model_id'] = SDIFF_LOCAL_MODELS['sd20_base']
    elif config['model']['model_id'] == 'stabilityai/stable-diffusion-2-inpainting':
        config['model']['model_id'] = SDIFF_LOCAL_MODELS['sd20_inpaint']
    return config


def load_pipeline(config, ckpt_path, outpaint: bool = False):
    model_class = PanoOutpaintor if outpaint else PanoGenerator
    model_ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
    model_ckpt = rename_attention_keys(model_ckpt)
    
    pipeline = model_class(config)
    pipeline.load_state_dict(model_ckpt, strict=False) # ignore "text_encoder.text_model.embeddings.position_ids"
    pipeline.eval()
    pipeline = pipeline.to(device=shared.device)
    # pipeline.to(dtype=shared.dtype)

    if not shared.low_vram:
        return pipeline

    print("\n\n Enabling memory efficiency ...")

    # Enable memory efficiency
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()

    # FIXME: attention_head_dim is a list, not int as SD-15
    attention_size = pipeline.mv_base_model.unet.config.attention_head_dim
    if isinstance(attention_size, int):
        slice_size = attention_size // 2
        pipeline.mv_base_model.unet.set_attention_slice(slice_size)

    # FIXME: some ops doesn't support FP16
    # if is_xformers_available():
    #     import xformers
    #     attention_op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
    #     pipeline.mv_base_model.unet.enable_xformers_memory_efficient_attention(attention_op)

    # FIXME: some ops doesn't support FP16
    # On/Off-loading
    # offload_config = dict(
    #     non_blocking = True,
    #     onload_device = shared.device, 
    #     offload_device = torch.device('cpu'), 
    # )

    # FIXME: customized pipeline doesn't auto-handle `group offloading`
    # apply_group_offloading(pipeline.vae, offload_type="leaf_level", **offload_config)
    # apply_group_offloading(pipeline.text_encoder, 
    #                                     offload_type="block_level", 
    #                                     num_blocks_per_group=2, **offload_config)
    # apply_group_offloading(pipeline.mv_base_model, 
    #                                     offload_type="leaf_level", 
    #                                     use_stream=True, 
    #                                     low_cpu_mem_usage=True, **offload_config)
    return pipeline


def run_pipeline(
    pipe,
    prompt,
    images_mv = None,
    guidance_scale = 7.7,
    diffusion_steps = 25,
    resolution = 512,
    num_views = 8,
    **kwargs
):

    images = torch.zeros((1, num_views, resolution, resolution, 3))
    if isinstance(images_mv, np.ndarray):
        images_mv = torch.tensor(images_mv)
    if isinstance(images_mv, torch.Tensor):
        images_mv = [(images_mv, 0)]
    if isinstance(images_mv, (list, tuple)):
        for img_sv, v in images_mv:
            if (v+1) > num_views:
                print(f"[WARNING] Ignore view = {v+1} > {num_views}")
                continue
            images[0, v] = (img_sv if isinstance(img_sv, torch.Tensor) 
                                else torch.tensor(img_sv))
    images = images.to(device=shared.device)

    if isinstance(prompt, str):
        prompt = [prompt] * num_views
    else:
        assert isinstance(prompt, (list, tuple))
        assert len(prompt) == num_views

    Rs, Ks = multiview_Rs_Ks(resolution, num_views)
    Ks = torch.tensor(Ks)[None].to(device=shared.device)
    Rs = torch.tensor(Rs)[None].to(device=shared.device)

    batch = {
        'images': images,
        'prompt': prompt,
             'R': Rs,
             'K': Ks,
    }

    if guidance_scale > 0:
        pipe.guidance_scale = guidance_scale
    
    if diffusion_steps > 0:
        pipe.diff_timestep = diffusion_steps

    generated = pipe.inference(batch)[0]
    return generated


if __name__ == "__main__":

    outpaint = True
    resolution = 512
    num_views = 8

    # mv2pano
    config_path = str(current_workdir / 'configs/pano_outpainting.yaml')
    ckpt_path = MVDIFF_LOCAL_MODELS["sd2_outpaint/mvdiffusion"]

    # Load pipeline
    clear_torch_cache()
    config = load_config(config_path)
    pipe = load_pipeline(config, ckpt_path, outpaint = outpaint)

    # Conditions
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

    image_mv_list = [
        ("C:/Users/Mr. RIAH/Documents/Projects/SHIVA/temp/multi_pov/mvdiff_outpaint_25_view01.png", 0),
        ("C:/Users/Mr. RIAH/Documents/Projects/SHIVA/temp/multi_pov/mvdiff_outpaint_25_view02.png", 1),
        ("C:/Users/Mr. RIAH/Documents/Projects/SHIVA/temp/multi_pov/mvdiff_outpaint_25_view03.png", 2),
        ("C:/Users/Mr. RIAH/Documents/Projects/SHIVA/temp/multi_pov/mvdiff_outpaint_25_view08.png", 7),
    ]

    images_mv = []
    for image_path, view_id in image_mv_list:
        image = cv2.imread(image_path)
        image = preprocess_image(image, resolution)
        image = torch.tensor(image)
        images_mv.append((image, view_id))

    # Inference
    for diff_steps in [10, 25]:
        gen_params = dict(
            num_views = num_views,
            resolution = resolution,
            guidance_scale = 9.5,
            diffusion_steps = diff_steps,
        )

        generated = run_pipeline(pipe, prompt, images_mv, **gen_params)

        # Save output
        print('\nSaving outputs ...')
        prefix = 'mvdiff_' + ('outpaint' if outpaint else 'generate') + f'_{diff_steps}_'
        image_paths = []
        for i in range(num_views):
            image_path = f"./temp/{prefix}view{i+1:02d}.png"
            image_paths.append(image_path)
            im = Image.fromarray(generated[i])
            im.save(image_path)

        generate_video(image_paths, './temp', prefix)

