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
        from diffusers import StableDiffusionImg2ImgPipeline as RefinementPipeline

    elif model_version == 'sdxl':
        from diffusers import StableDiffusionXLImg2ImgPipeline as RefinementPipeline

    else:
        raise ValueError(f"{model_version} is not supported!")

    config = dict(torch_dtype=shared.dtype, local_files_only=False)

    if model_version.startswith('sd'):
        config['num_in_channels'] = 9 if model_name.endswith('inpaint') else 4

    if shared.low_vram:
        config['low_cpu_mem_usage'] = True 

    model_path = str(shared.IMAGENE_LOCAL_MODELS[f"{model_version}/{model_name}"])
    if model_path.endswith(shared.model_extensions):
        config.update(dict(use_safetensors=True if model_path.endswith(".safetensors") else False))
        pipe = RefinementPipeline.from_single_file(model_path, **config).to(shared.device)
    else:
        pipe = RefinementPipeline.from_pretrained(model_path, **config).to(shared.device)
    return pipe


def run_pipeline(
    pipe, 
    image: ImageClass,
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
        progress_bar.set_description(f"Refinement {i+1} / {batch_size} ")
        generated = pipe(image = image,
                        prompt = prompt, 
                negative_prompt = nrompt, **diffusion_kwargs).images[0]
        all_generated.append(generated)
    return all_generated


if __name__ == "__main__":

    image = Image.open("C:/Users/Mr. RIAH/Pictures/Shinhan-06.png").convert("RGB")
    W, H = image.size

    prompt = 'a promotional booth with "Shinhan Bank" signage, blue LED backlit, photorealistic'

    ############################################################
    #                       Load Pipeline                      #
    ############################################################

    # model_selected = "sd15/dreamshaper_v8"
    model_selected = "sdxl/sdxl_refine_v1"
    model_version, model_name = model_selected.split('/')

    pipe = load_pipeline(model_name, model_version)

    # enable memory savings
    if shared.low_vram:
        pipe = enable_lowvram_usage(pipe)

    ###########################################################
    #                       Run Pipeline                      #
    ###########################################################

    from ..default import POSITIVE_PROMPT, NEGATIVE_PROMPT

    prompt = POSITIVE_PROMPT + ', ' + prompt
    nrompt = NEGATIVE_PROMPT

    config = dict(
        height = H * 2, 
        width = W * 2,
        strength = 0.33, 
        guidance_scale = 4.9, 
    num_inference_steps = 25, 
        output_type = 'pil',
    )

    image = run_pipeline(pipe, image, prompt, nrompt, **config)[0]
    image.save(f'./temp/refined_{model_version}.png')

