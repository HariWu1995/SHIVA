from pathlib import Path
from typing import List, Union
from tqdm import tqdm
from PIL import Image
from PIL.Image import Image as ImageClass

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:512"

import cv2
import numpy as np

import torch
from diffusers import AutoencoderKL, ControlNetModel
from diffusers import StableDiffusionXLControlNetPipeline as StylizationPipeline

from . import shared
from .path import SDIFF_LOCAL_MODELS, STYLE_LOCAL_MODELS, LOCAL_IMAGE_ENCODERS
from .utils import clear_torch_cache, resize_image


#################################
#       Extra source code       #
#################################

extra_lib = str(Path(__file__).resolve().parents[4] / 'extra')

import sys
sys.path.append(extra_lib)

from ip_adapter import IPAdapterXL_CSGO
from ip_adapter.utils import resize_content


#################################
#       Global variables        #
#################################

UNET_BLOCKS = {
    'content': ['down_blocks'],
    'style': ["up_blocks"],
}

CTRLNET_BLOCKS = {
    'content': [],
    'style': ["down_blocks"],
}

unet_target_content_blocks = UNET_BLOCKS['content']
unet_target_style_blocks   = UNET_BLOCKS['style']
ctrl_target_content_blocks = CTRLNET_BLOCKS['content']
ctrl_target_style_blocks   = CTRLNET_BLOCKS['style']


POSITIVE_PROMPT = 'best quality, 4k resolution, hires, highly detailed'
NEGATIVE_PROMPT = 'text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry'


#################################
#           Functions           #
#################################

def load_pipeline(
    base_model: str, 
    ctrl_model: str = "sdxl/csgo_ctrl",
    adapter_name: str = "sdxl/csgo_4_32",
    encoder_name: str = "clip-vit-bigg14",
):

    assert base_model.startswith('sdxl')
    assert adapter_name in ['sdxl/csgo', 'sdxl/csgo_4_32']

    if adapter_name == 'sdxl/csgo':
        num_content_tokens = 4
        num_style_tokens = 16

    elif adapter_name == 'sdxl/csgo_4_32':
        num_content_tokens = 4
        num_style_tokens = 32

    clear_torch_cache()

    model_path = SDIFF_LOCAL_MODELS[base_model]
    ctrlnet_path = STYLE_LOCAL_MODELS[ctrl_model]
    adapter_path = STYLE_LOCAL_MODELS[adapter_name]
    encoder_path = LOCAL_IMAGE_ENCODERS[encoder_name]

    controlnet = ControlNetModel.from_pretrained(ctrlnet_path, torch_dtype=shared.dtype, use_safetensors=True)

    config = dict(
        controlnet=controlnet,
        torch_dtype=shared.dtype, 
        local_files_only=False,
        add_watermarker=False,
    )

    if shared.low_vram:
        config['low_cpu_mem_usage'] = True 

    if model_path.endswith(shared.model_extensions):
        config.update(dict(use_safetensors=True if model_path.endswith(".safetensors") else False))
        pipe = StylizationPipeline.from_single_file(model_path, **config).to(shared.device)
    else:
        pipe = StylizationPipeline.from_pretrained(model_path, **config).to(shared.device)

    # Enable low VRAM strategy
    pipe.enable_vae_tiling()

    # Load adaptor
    pipe_cs = IPAdapterXL_CSGO(
        pipe, 
        encoder_path, 
        adapter_path, 
        shared.device, 
        num_content_tokens=num_content_tokens,
        num_style_tokens=num_style_tokens,
        target_content_blocks=unet_target_content_blocks, 
        target_style_blocks=unet_target_style_blocks,
        controlnet_adapter=True,
        controlnet_target_content_blocks=ctrl_target_content_blocks,
        controlnet_target_style_blocks=ctrl_target_style_blocks,
        content_model_resampler=True,
        style_model_resampler=True,
    )

    return pipe_cs


def run_pipeline(
    pipe, 
    stylize_image: ImageClass,
    content_image: ImageClass | None = None,
    prompt: str = POSITIVE_PROMPT, 
    nrompt: str = NEGATIVE_PROMPT, 
    batch_size: int = 1,
    diffusion_steps: int = 25,
    content_scale: float = 1.0,
    stylize_scale: float = 1.0,
    guidance_scale: float = 9.5,
    controlnet_scale: float = 0.69,
    **kwargs
):
    if content_image is None:
        content_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        content_image = Image.fromarray(content_image)
        content_scale = 0.5

    content_image = content_image.convert('RGB')
    width, height, content_image = resize_image(content_image)

    diffusion_kwargs = dict(
            style_scale=stylize_scale,
            content_scale=content_scale,
            guidance_scale=guidance_scale,
controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=diffusion_steps,
            num_images_per_prompt=1, 
            num_samples=1,
    )
    diffusion_kwargs.update(kwargs)

    all_generated = []
    progress_bar = tqdm(list(range(batch_size)))
    for i in progress_bar:
        progress_bar.set_description(f"Generating {i+1} / {batch_size} ")
        generated = pipe.generate(prompt = prompt, 
                         negative_prompt = nrompt, 
                          pil_style_image = stylize_image, 
                        pil_content_image = content_image, 
                                    image = content_image,
                                    width = width, 
                                    height = height, **diffusion_kwargs).images[0]
        all_generated.append(generated)
    return all_generated


if __name__ == "__main__":

    #####################################
    #           Load Pipeline           #
    #####################################

    # model_name = "sdxl/sdxl_base_v1"
    model_name = "sdxl/dreamshaper_light"

    pipe = load_pipeline(model_name)

    #####################################
    #            Run Pipeline           #
    #####################################

    prompt = "car showroom, glossy floor reflecting the soft lighting, daylight, polished surface, large windows, city view, minimalist modern design"
    prompt = POSITIVE_PROMPT + ', ' + prompt
    nrompt = NEGATIVE_PROMPT

    stylize_image = Image.open("C:/Users/Mr. RIAH/Documents/Projects/Shinhan/Booths/z6512519123413_fa23e5b854d39565db83ee725fcc3a9a.jpg")
    content_image = Image.open("C:/Users/Mr. RIAH/Pictures/_booth/booth-24.png")

    stylize_image.save(f'./temp/csgo_style.png')
    content_image.save(f'./temp/csgo_content.png')

    config = dict(guidance_scale = 9.5, diffusion_steps = 20)

    image = run_pipeline(pipe, stylize_image, content_image, prompt, nrompt, **config)[0]
    image.save(f'./temp/csgo_generated.png')

