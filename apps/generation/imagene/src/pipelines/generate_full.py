from typing import List, Union
from tqdm import tqdm

from PIL import Image
from PIL.Image import Image as ImageClass

import numpy as np

from .. import shared
from ..utils import clear_torch_cache
from .utils import enable_lowvram_usage


def load_pipeline(
    model_name: str, 
    model_version: str, 
    num_in_channels: int = 9,
):
    clear_torch_cache()

    if model_version == 'sd15':
        from diffusers import StableDiffusionPipeline as GenerationPipeline

    elif model_version == 'sdxl':
        from diffusers import StableDiffusionXLPipeline as GenerationPipeline

    elif model_version == 'flux':
        from diffusers import FluxPipeline as GenerationPipeline

    config = dict(torch_dtype=shared.dtype, local_files_only=False)

    if model_version.startswith('sd'):
        config['num_in_channels'] = num_in_channels

    if shared.low_vram:
        config['low_cpu_mem_usage'] = True 

    model_path = str(shared.IMAGENE_LOCAL_MODELS[f"{model_version}/{model_name}"])
    if model_path.endswith(shared.model_extensions):
        config.update(dict(use_safetensors=True if model_path.endswith(".safetensors") else False))
        pipe = GenerationPipeline.from_single_file(model_path, **config).to(shared.device)
    else:
        pipe = GenerationPipeline.from_pretrained(model_path, **config).to(shared.device)

    # enable memory savings
    if shared.low_vram:
        pipe = enable_lowvram_usage(pipe)
    return pipe


def run_pipeline(
    model_name: str, 
    model_version: str, 
    prompt: str = '', 
    nrompt: str = '', 
    batch_size: int = 1,
    **kwargs
):

    n_channels = 9 if model_name.endswith('inpaint') else 4
    pipe = load_pipeline(model_name, model_version, n_channels)

    if model_version == "sd15":
        H, W = 512, 512
    elif model_version == "sdxl":
        H, W = 1024, 1024
    elif model_version == "flux":
        H, W = 1024, 1024

    diffusion_kwargs = dict(height = H, width = W)
    diffusion_kwargs.update(kwargs)

    all_generated = []
    progress_bar = tqdm(list(range(batch_size)))
    for i in progress_bar:
        progress_bar.set_description(f"Generating {i+1} / {batch_size} ")
        generated = pipe(prompt = prompt, 
                negative_prompt = nrompt, **diffusion_kwargs).images[0]
        all_generated.append(generated)
    return all_generated


if __name__ == "__main__":

    from ..utils import POSITIVE_PROMPT, NEGATIVE_PROMPT

    # model_selected = "sd15/dreamshaper_v8"
    model_selected = "sdxl/dreamshaper_light"
    model_version, model_name = model_selected.split('/')
    model_channels = 9 if model_name.endswith('inpaint') else 4

    prompt = "car showroom, glossy floor reflecting the soft lighting, daylight, polished surface, large windows, city view, minimalist modern design"
    prompt = POSITIVE_PROMPT + ', ' + prompt
    nrompt = NEGATIVE_PROMPT

    config = dict(strength=0.9, guidance_scale=7.7, num_inference_steps=10, output_type='pil')
    image = run_pipeline(model_name, model_version, prompt, nrompt, **config)[0]
    image.save(f'./temp/generated_{model_version}.png')

