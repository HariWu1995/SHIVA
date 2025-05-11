from PIL import Image

import torch
from diffusers import EulerAncestralDiscreteScheduler as BaseScheduler

from .. import shared
from .utils import enable_lowvram_usage
from .diffusion360.txt2pano import Text2PanoBaseline


positive_prompt = 'photorealistic, realistic textures, artstation, best quality, ultra res, 8k'

negative_prompt = 'worst quality, low quality, logo, text, watermark, monochrome, blur, blurry'\
                ', deformed, disfigured, distorted, messy, unnatural overlap, tangled patterns'\
                ', human, person, small object, (pillar), (barriers), closed space'\
                ', complex texture, complex lighting, oversaturated'


def load_model(model_id: str = "sd15/diffusion360"):
    model_path = shared.MVDIFF_LOCAL_MODELS[model_id] + '/sd-base'
    model = Text2PanoBaseline.from_pretrained(model_path, torch_dtype=shared.dtype).to(device=shared.device)
    model.scheduler = BaseScheduler.from_config(model.scheduler.config)

    if shared.low_vram:
        model = enable_lowvram_usage(model)
    return model


def inference(
    model, 

    positive_prompt: str,
    negative_prompt: str,

    image_size: int = 768,
    diffusion_steps: int = 20, 
    guidance_scale: float = 9.5, 

    **kwargs
):
    height = image_size
    width = image_size * 2

    if '<360panorama>' not in positive_prompt:
        positive_prompt = f'<360panorama>, {positive_prompt}'

    gen_params = dict(
         prompt = positive_prompt,
negative_prompt = negative_prompt,
        width = width, 
        height = height, 
        guidance_scale = guidance_scale,
        num_inference_steps = diffusion_steps,
    )

    output = model(**gen_params).images[0]
    return output


if __name__ == "__main__":

    # Inputs
    prompt = "modern exhibition hall, multiple branded booths, digital displays, "\
            "promotional banners, interactive kiosks, LED screens, elegant decor, "\
            "symmetrical layout, wooden wall, wooden floor, high ceiling"

    # Preprocess
    positive_prompt = prompt + ', ' + positive_prompt
    output_name = 'diffusion360_txt2pano'

    # Load pipeline
    model = load_model()
    
    # Inference
    output = inference(
        model, 
        positive_prompt,
        negative_prompt,
        image_size=768,
    )
    output.save(f'./temp/{output_name}_x1.png')

