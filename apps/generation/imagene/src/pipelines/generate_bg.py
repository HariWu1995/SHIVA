from pathlib import Path
from typing import List, Union
from tqdm import tqdm

from PIL import Image
from PIL.Image import Image as ImageClass

import numpy as np

from .. import shared
from ..utils import clear_torch_cache
from .utils import enable_lowvram_usage, blend_images

# Link to BrushNet
import sys
sys.path.append(Path(__file__).resolve().parents[5] / 'extra')


def load_pipeline(
    model_name: str, 
    model_version: str, 
    num_in_channels: int = 9,
    brushnet_name: str | None = None,
):
    clear_torch_cache()

    if (model_version == 'sd15') and (brushnet_name is not None):
        from brushnet import StableDiffusionBrushNetPipeline as InpaintPipeline, BrushNetModel

    elif (model_version == 'sdxl') and (brushnet_name is not None):
        from brushnet import StableDiffusionXLBrushNetPipeline as InpaintPipeline, BrushNetModel

    elif model_version == 'sd15':
        from diffusers import StableDiffusionInpaintPipeline as InpaintPipeline

    elif model_version == 'sdxl':
        from diffusers import StableDiffusionXLInpaintPipeline as InpaintPipeline

    elif model_version == 'flux':
        if model_name.endswith('fill'):
            from diffusers import FluxFillPipeline as InpaintPipeline
        else:
            from diffusers import FluxInpaintPipeline as InpaintPipeline

    else:
        raise ValueError(f"{model_version} is not supported!")

    config = dict(torch_dtype=shared.dtype, local_files_only=False)
    
    if model_version.startswith('sd'):
        config['num_in_channels'] = num_in_channels

    if brushnet_name is not None:
        brushnet_path = str(shared.IMAGENE_LOCAL_MODELS[f"brush/{brushnet_name}"])
        brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=shared.dtype)
        config['brushnet'] = brushnet

    model_path = str(shared.IMAGENE_LOCAL_MODELS[f"{model_version}/{model_name}"])
    if model_path.endswith(shared.model_extensions):
        config.update(dict(use_safetensors=True if model_path.endswith(".safetensors") else False))
        pipe = InpaintPipeline.from_single_file(model_path, **config).to(shared.device)
    else:
        pipe = InpaintPipeline.from_pretrained(model_path, **config).to(shared.device)

    # enable memory savings
    if shared.low_vram:
        pipe = enable_lowvram_usage(pipe)
    return pipe


def run_pipeline(
    model_name: str, 
    model_version: str, 
    image: ImageClass, 
    mask: ImageClass, 
    prompt: str = '', 
    nrompt: str = '', 
    n_channels: int = 9,    # inpaint = 9 channels
    batch_size: int = 1,
    brushnet_name: str | None = None,
    brushnet_scale: float = 1.0,
    blend_after_gen: bool = False,
    **kwargs
):

    pipe = load_pipeline(model_name, model_version, n_channels)
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    
    W, H = image.size

    diffusion_kwargs = dict(height = H, width = W)
    diffusion_kwargs.update(kwargs)

    if brushnet_name is not None:
        diffusion_kwargs['brushnet_conditioning_scale'] = brushnet_scale

    all_generated = []
    progress_bar = tqdm(list(range(batch_size)))
    for i in progress_bar:
        progress_bar.set_description(f"Generating {i+1} / {batch_size} ")
        generated = pipe(prompt = prompt, 
                negative_prompt = nrompt, 
                         image = image, 
                    mask_image = mask, **diffusion_kwargs).images[0]
        all_generated.append(generated)
    return all_generated


if __name__ == "__main__":

    from ..utils import POSITIVE_PROMPT, NEGATIVE_PROMPT

    # model_selected = "sd15/dreamshaper_v8_inpaint"
    model_selected = "sdxl/dreamshaper_light_inpaint"
    model_version, model_name = model_selected.split('/')
    model_channels = 9 if model_name.endswith('inpaint') else 4

    # brushnet = None
    brushnet = "random_mask"
    if model_version == 'sdxl':
        brushnet += '_xl'

    image = Image.open("C:/Users/Mr. RIAH/Documents/GenAI/_Visual/Anilluminus.AI/logs/image.png")
    mask  = Image.open("C:/Users/Mr. RIAH/Documents/GenAI/_Visual/Anilluminus.AI/logs/mask.png")
    
    prompt = "car showroom, glossy floor reflecting the soft lighting, daylight, polished surface, large windows, city view, minimalist modern design"
    prompt = POSITIVE_PROMPT + ', ' + prompt
    nrompt = NEGATIVE_PROMPT

    config = dict(strength=0.9, guidance_scale=7.7, num_inference_steps=30, output_type='pil')
    if brushnet is not None:
        config['brushnet_name'] = brushnet

    save_prefix = 'inpainted' if not brushnet else 'brushpainted'

    imgenerated = run_pipeline(model_name, model_version, image, mask, prompt, nrompt, **config)[0]
    imgenerated.save(f'./temp/{save_prefix}_{model_version}.png')

    imgen_blended = blend_images(image, imgenerated, mask)
    imgen_blended.save(f'./temp/{save_prefix}_{model_version}_blended.png')

