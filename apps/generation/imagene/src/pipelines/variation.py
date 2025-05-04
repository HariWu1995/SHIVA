"""
Image Variation using IP-Adapter:
    https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter
"""
from pathlib import Path
from typing import List, Union
from tqdm import tqdm
from os import path as osp

from PIL import Image
from PIL.Image import Image as ImageClass

import numpy as np
import torch

from ..utils import clear_torch_cache
from .utils import enable_lowvram_usage
from .. import shared


def load_pipeline(
    model_name: str, 
    model_version: str, 
    adapter_name: str = 'ip_adapter', 
):
    clear_torch_cache()

    if model_version == 'sd15':
        from diffusers import StableDiffusionPipeline as VariationPipeline

    elif model_version == 'sdxl':
        from diffusers import StableDiffusionXLPipeline as VariationPipeline

    else:
        raise ValueError(f"{model_version} is not supported!")

    config = dict(torch_dtype=shared.dtype, local_files_only=False, safety_checker=None)

    if shared.low_vram:
        config['low_cpu_mem_usage'] = True 

    model_path = str(shared.IMAGENE_LOCAL_MODELS[f"{model_version}/{model_name}"])
    if model_path.endswith(shared.model_extensions):
        config.update(dict(use_safetensors=True if model_path.endswith(".safetensors") else False))
        pipe = VariationPipeline.from_single_file(model_path, **config).to(shared.device)
    else:
        pipe = VariationPipeline.from_pretrained(model_path, **config).to(shared.device)

    # Load IP-Adapter
    iencoder_name = "clip-vit-" + ("h14" if model_version == 'sd15' else "bigg14")
    iencoder_path = str(shared.LOCAL_IMAGE_ENCODERS[iencoder_name])
    adapter_path = str(shared.LOCAL_IP_ADAPTERS[f"{model_version}/ip/{adapter_name}"])
    adapter_dir = osp.dirname(adapter_path)
    adapter_fn = osp.basename(adapter_path)

    pipe.load_ip_adapter(
        adapter_dir, 
        subfolder='',
        weight_name=adapter_fn,
        image_encoder_folder=iencoder_path,
    )
    return pipe


def run_pipeline(
    pipe, 
    image: ImageClass, 
    prompt: str = shared.POSITIVE_PROMPT, 
    nrompt: str = shared.NEGATIVE_PROMPT, 
    batch_size: int = 1,
    adapt_scale: float = 0.6,    # Adjust between 0.0 (text only) and 1.0 (image only)
    **kwargs
):

    pipe.set_ip_adapter_scale(adapt_scale)

    diffusion_kwargs = dict(prompt=prompt, negative_prompt=nrompt)
    diffusion_kwargs.update(kwargs)

    all_generated = []
    progress_bar = tqdm(list(range(batch_size)))
    for i in progress_bar:
        progress_bar.set_description(f"Generating {i+1} / {batch_size} ")
        clear_torch_cache()
        generated = pipe(ip_adapter_image = image, **diffusion_kwargs).images[0]
        all_generated.append(generated)
    return all_generated


if __name__ == "__main__":

    ############################################################
    #                       Load Pipeline                      #
    ############################################################

    model_selected = "sd15/dreamshaper_v8"
    # model_selected = "sdxl/dreamshaper_light"
    model_version, model_name = model_selected.split('/')

    adapter_name = "ip_adapter"

    pipe = load_pipeline(model_name, model_version, adapter_name)

    # enable memory savings
    if shared.low_vram:
        pipe = enable_lowvram_usage(pipe, offload_only=True)

    ###########################################################
    #                       Run Pipeline                      #
    ###########################################################

    image = Image.open("C:/Users/Mr. RIAH/Pictures/_stage/stage-04.png")
    # image = Image.open("C:/Users/Mr. RIAH/Pictures/booth_smart_home_device.jpg")
    # image = Image.open("C:/Users/Mr. RIAH/Pictures/_gate/welcome-gate-001-M.png")
    # image = Image.open("C:/Users/Mr. RIAH/Pictures/__harievent/( Anhpng.com ) - Decor-chup-anh-2025 - 05 (Custom).jpg")
    image = image.convert('RGB')
    image.save(f'./temp/variated_{model_version}_00.png')

    W, H = image.size

    description = "a stage, three semi-circular backdrops, pastel purple, light peach, light blue, abstract fluid paint design, metal stage truss, spotlights, dark background, photorealistic rendering, 8k resolution"
    # description = "modern exhibition booth design, showcasing home appliances, white washing machines, marble surfaces. accent lighting highlights the products and architectural features. hanging floral installations add a touch of elegance. highly detailed, realistic rendering, 8k resolution, yellow lighting."
    # description = "a large archway entrance to a job fair, detailed cosmic background with nebulae, planets, and a futuristic cityscape, Shinhan logo prominently displayed, blue carpet leading to the arch, neon lighting, corporate event, hyperrealistic, 8k resolution, cinematic lighting"
    # description = "a stage, 2-layered backdrop celebrating festive Vietnamese Tet 2025, red theme. A friendly snake mascot holding gold ingots. Red lanterns, a lion dance mask, plum blossoms, and firecrackers. Vietnamese calligraphy celebrating the New Year. detailed textures, sharp lines, 8k resolution"

    config = dict(
        height = 768, 
        width = 1024,
        adapt_scale = 0.55,
        guidance_scale = 7.7, 
    num_inference_steps = 33, 
        output_type = 'pil',
        batch_size = 4,
        prompt = description,
    )

    images = run_pipeline(pipe, image, **config)
    for i, img in enumerate(images):
        img.save(f'./temp/variated_{model_version}_{i+1:02d}.png')

