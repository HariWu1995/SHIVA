from pathlib import Path
from typing import List, Union
from tqdm import tqdm

from PIL import Image
from PIL.Image import Image as ImageClass

import numpy as np
import torch

from .. import shared
from ..utils import clear_torch_cache
from .utils import enable_lowvram_usage

# Link to Extra
extra_lib = str(Path(__file__).resolve().parents[5] / 'extra')

import sys
sys.path.append(extra_lib)


def load_pipeline(
    model_name: str, 
    model_version: str, 
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
        config['num_in_channels'] = 9 if model_name.endswith('inpaint') else 4

    if shared.low_vram:
        config['low_cpu_mem_usage'] = True 

    model_path = str(shared.IMAGENE_LOCAL_MODELS[f"{model_version}/{model_name}"])
    if (model_version == 'flux') and (model_name == 'mini'):
        from flux_mini import load_pipeline as load_pipeline_flux_mini
        pipe = load_pipeline_flux_mini(model_path)
    elif model_path.endswith(shared.model_extensions):
        config.update(dict(use_safetensors=True if model_path.endswith(".safetensors") else False))
        pipe = GenerationPipeline.from_single_file(model_path, **config).to(shared.device)
    else:
        pipe = GenerationPipeline.from_pretrained(model_path, **config).to(shared.device)
    return pipe


def run_pipeline(
    pipe, 
    prompt: str = '', 
    nrompt: str = '', 
    batch_size: int = 1,
    **kwargs
):

    diffusion_kwargs = dict()
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

    ############################################################
    #                       Load Pipeline                      #
    ############################################################

    # model_selected = "flux/mini"
    model_selected = "sd15/dreamshaper_v8"
    # model_selected = "sdxl/dreamshaper_light"
    model_version, model_name = model_selected.split('/')

    pipe = load_pipeline(model_name, model_version)

    # enable memory savings
    if shared.low_vram:
        pipe = enable_lowvram_usage(pipe)

    if model_version == "sd15":
        H, W = 512, 512
    elif model_version == "sdxl":
        H, W = 1024, 1024
    elif model_version == "flux":
        H, W = 1024, 1024

    ###########################################################
    #                       Run Pipeline                      #
    ###########################################################

    from ..default import POSITIVE_PROMPT, NEGATIVE_PROMPT

    prompt = "car showroom, glossy floor reflecting the soft lighting, daylight, polished surface, large windows, city view, minimalist modern design"
    prompt = POSITIVE_PROMPT + ', ' + prompt
    nrompt = NEGATIVE_PROMPT

    config = dict(
        height = H, 
        width = W,
        strength = 0.9, 
        guidance_scale = 7.7, 
    num_inference_steps = 30, 
        output_type = 'pil',
    )

    image = run_pipeline(pipe, prompt, nrompt, **config)[0]
    image.save(f'./temp/generated_{model_version}.png')

