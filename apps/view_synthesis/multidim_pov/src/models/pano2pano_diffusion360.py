from pathlib import Path
from PIL import Image

import math
import torch
from diffusers import ControlNetModel
from diffusers.utils import load_image

from .. import shared
from .utils import enable_lowvram_usage
from .diffusion360.img2pano import Image2PanoControline


positive_prompt = 'photorealistic, realistic textures, artstation, best quality, ultra res'\
                ', 8k rendering, wide angle view, immersive scenery, immense, vast extent'

negative_prompt = 'worst quality, low quality, logo, text, watermark, monochrome, blur, blurry'\
                ', deformed, disfigured, distorted, messy, unnatural overlap, tangled patterns'\
                ', human, person, small object, (pillar), (barriers), closed space'\
                ', complex texture, complex lighting, oversaturated'


def load_model(model_id: str = "sd15/diffusion360"):
    model_dir = shared.MVDIFF_LOCAL_MODELS[model_id]
    ctrl_path = model_dir + '/sr-control'
    model_path = model_dir + '/sr-base'

    controlnet = ControlNetModel.from_pretrained(ctrl_path, torch_dtype=shared.dtype)
    model = Image2PanoControline.from_pretrained(model_path, torch_dtype=shared.dtype,
                                        controlnet=controlnet).to(device=shared.device)
    # model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)

    if shared.low_vram:
        model = enable_lowvram_usage(model)
    return model


def inference(
    model, 

    positive_prompt: str,
    negative_prompt: str,

    image: Image.Image,
    upscale: int = 2,
    generator = None,

    strength: float = 0.5,
    diffusion_steps: int = 10, 
    guidance_scale: float = 7., 
    controlnet_scale: float = 1.,

    **kwargs
):
    w, h = image.size
    w_fit = math.ceil(w / 768) * 768
    h_fit = math.ceil(h / 768) * 768
    upscale = int(upscale)

    if upscale > 0:
        width  = w_fit * upscale
        height = h_fit * upscale
        image = image.resize((width, height))
    else:
        image = image.resize((w_fit, h_fit))

    gen_params = dict(
         prompt = positive_prompt,
negative_prompt = negative_prompt,
        width = width, 
        height = height, 
        strength = strength,
        image = image,
control_image = image, 
controlnet_conditioning_scale = controlnet_scale,
        guidance_scale = guidance_scale,
        num_inference_steps = diffusion_steps,
        generator = generator,
    )

    output = model(**gen_params).images[0]
    return output


if __name__ == "__main__":

    # Inputs
    # img_path = "C:/Users/Mr. RIAH/Documents/Projects/SHIVA/temp/diffusion360_img2pano_x1.png"
    # upscale = 2
    # prompt = "((a backdrop of a launching event)) at an extended exhibition hall, "\
    #         "open-ended long lobby on two sides, elegant minimal interior, "\
    #         "symmetrical layout, wooden wall, wooden floor, high ceiling"

    img_path = "C:/Users/Mr. RIAH/Documents/Projects/SHIVA/temp/mvdiff_outpaint_25_pano.png"
    upscale = 2
    prompt = "modern conference stage with central LED screen and two side screens, "\
             "rows of white-covered tables and chairs, overhead spotlights"

    image = load_image(img_path)
    image = image.resize((2048, 512))

    # Preprocess
    positive_prompt = prompt + ', ' + positive_prompt
    # output_name = 'diffusion360_pano2pano'
    output_name = 'mvdiff_outpaint_25_pano'

    # Load pipeline
    model = load_model()
    
    # Inference
    output = inference(
        model, 
        positive_prompt,
        negative_prompt,
        image, 
        upscale=upscale,
    )
    output.save(f'./temp/{output_name}_x2.png')

