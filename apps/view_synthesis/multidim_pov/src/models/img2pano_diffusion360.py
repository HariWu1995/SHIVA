from PIL import Image

import torch
from diffusers.utils import load_image

from .. import shared
from .utils import resize_and_center_crop
from .diffusion360.img2pano import Image2PanoPipeline


positive_prompt = 'photorealistic, realistic textures, artstation, best quality, ultra res'\
                ', 8k rendering, wide angle view, immersive scenery, immense, vast extent'

negative_prompt = 'worst quality, low quality, logo, text, watermark, monochrome, blur, blurry'\
                ', deformed, disfigured, distorted, messy, unnatural overlap, tangled patterns'\
                ', human, person, small object, (pillar), (barriers), closed space'\
                ', complex texture, complex lighting, oversaturated'


def load_and_resize_image(image_path, size: int = 768):
    image = load_image(image_path)
    image = image.resize((768, 768))
    image = resize_and_center_crop(image, 768)
    return image


def load_model(model_id: str = "sd15/diffusion360", refine: bool = True):
    model_path = shared.MVDIFF_LOCAL_MODELS[model_id]
    model = Image2PanoPipeline(model_path, refine=refine)
    return model


def inference(
    model, 

    positive_prompt: str,
    negative_prompt: str,

    image: Image.Image,
    mask: Image.Image | None = None,

    image_size: int = 768,

    strength: float = 1.,
    diffusion_steps: int = 20, 
    guidance_scale: float = 9.5, 
    controlnet_scale: float = 1.,

    **kwargs
):
    height = image_size
    width = image_size * 2

    gen_params = dict(
        width = width, 
        height = height, 
        base_strength = strength,
        guidance_scale = guidance_scale,
        num_inference_steps = diffusion_steps,
        control_condition_scale = controlnet_scale,
    )

    output = model.generate(positive_prompt, negative_prompt, image, mask, **gen_params)
    return output


def enhancement(
    model, 

    positive_prompt: str,
    negative_prompt: str,

    image: Image.Image,
    upscale: int = 2,

    strength: float = 0.5,
    diffusion_steps: int = 10, 
    guidance_scale: float = 7.0, 
    controlnet_scale: float = 1.,

    **kwargs
):
    w, h = image.size
    assert (w % 768 == 0) and (h % 768 == 0)

    if upscale > 0:
        width = w * upscale
        height = h * upscale
        image = image.resize((width, height))

    gen_params = dict(
        refine_strength = strength,
        refine_scale = guidance_scale,
        num_refinement_steps = diffusion_steps,
        control_refine_scale = controlnet_scale,
    )

    output = model.refine(positive_prompt, negative_prompt, image, **gen_params)
    return output


if __name__ == "__main__":

    # Inputs
    prompt  = "((a backdrop of a launching event)) at an extended exhibition hall, "\
              "open-ended long lobby on two sides, elegant minimal interior, "\
              "symmetrical layout, wooden wall, wooden floor, high ceiling"

    image_path = "C:/Users/Mr. RIAH/Pictures/_gate/welcome-gate-001-L.png"
    image = load_and_resize_image(image_path, size=768)

    # Preprocess
    positive_prompt = prompt + ', ' + positive_prompt
    output_name = 'diffusion360_img2pano'

    # Load pipeline
    model = load_model()
    
    # Inference
    output = inference(
        model, 
        positive_prompt,
        negative_prompt,
        image,mask=None,
        image_size=768,
    )
    output.save(f'./temp/{output_name}_x1.png')

    full_scale = 1
    for scale in [2]:
        image = output
        output = enhancement(
            model, 
            positive_prompt,
            negative_prompt,
            image, 
            upscale=scale,
        )
        
        full_scale *= scale
        output.save(f'./temp/{output_name}_x{full_scale}.png')

