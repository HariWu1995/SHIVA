from pathlib import Path
from typing import List, Union
from tqdm import tqdm

from PIL import Image
from PIL.Image import Image as ImageClass

import numpy as np
import torch

from ..utils import clear_torch_cache, find_divisible
from .utils import enable_lowvram_usage
from .. import shared


def load_pipeline(
    model_name: str, 
    model_version: str, 
):
    clear_torch_cache()

    if model_version == 'sd15':
        from diffusers import StableDiffusionImageVariationPipeline as VariationPipeline
        from torchvision import transforms
        transf = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
                 transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                      [0.26862954, 0.26130258, 0.27577711]),
                ])

    elif model_version == 'flux':
        from diffusers import FluxPriorReduxPipeline as VariationPipeline
        transf = None

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

    # enable memory savings
    if shared.low_vram:
        pipe = enable_lowvram_usage(pipe)
    return pipe, transf


def run_pipeline(
    pipe, 
    image: ImageClass, 
    pretrans = None,
    batch_size: int = 1,
    **kwargs
):

    if pretrans is not None:
        image = pretrans(image).to(shared.device).unsqueeze(0)

    diffusion_kwargs = dict()
    diffusion_kwargs.update(kwargs)

    all_generated = []
    progress_bar = tqdm(list(range(batch_size)))
    for i in progress_bar:
        progress_bar.set_description(f"Generating {i+1} / {batch_size} ")
        generated = pipe(image = image, **diffusion_kwargs).images[0]
        all_generated.append(generated)
    return all_generated


if __name__ == "__main__":

    ############################################################
    #                       Load Pipeline                      #
    ############################################################

    # model_selected = "flux/mini"
    model_selected = "sd15/lambdalabs_img_variations"
    model_version, model_name = model_selected.split('/')

    pipe, pretrans = load_pipeline(model_name, model_version)

    ###########################################################
    #                       Run Pipeline                      #
    ###########################################################

    image = Image.open("C:/Users/Mr. RIAH/Pictures/_stage/stage-04.png")
    image = image.convert('RGB')
    image.save(f'./temp/variated_{model_version}_00.png')

    W, H = image.size

    config = dict(
        num_inference_steps = 25, 
        guidance_scale = 6.9, 
        output_type = 'pil',
        batch_size = 4,
    )

    images = run_pipeline(pipe, image, pretrans, **config)
    for i, img in enumerate(images):
        img.save(f'./temp/variated_{model_version}_{i+1:02d}.png')

