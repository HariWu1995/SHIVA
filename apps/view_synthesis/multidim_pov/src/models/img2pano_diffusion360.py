from pathlib import Path
from PIL import Image

import numpy as np
import py360convert as py360

import torch
from diffusers import ControlNetModel
from diffusers.utils import load_image

from .. import shared
from .utils import enable_lowvram_usage, resize_and_center_crop
from .diffusion360.img2pano import Image2PanoBaseline


positive_prompt = 'photorealistic, realistic textures, artstation, best quality, ultra res, 8k'

negative_prompt = 'worst quality, low quality, logo, text, watermark, monochrome, blur, blurry'\
                ', deformed, disfigured, distorted, messy, unnatural overlap, tangled patterns'\
                ', human, person, small object, (pillar), (barriers), closed space'\
                ', complex texture, complex lighting, oversaturated'


def load_and_resize_image(image_path, size: int = 768):
    image = load_image(image_path)
    image = image.resize((size, size))
    image = resize_and_center_crop(image, size)
    return image


def process_control_image(image, mask, width=1024, height=512):

    def to_tensor(img: Image.Image, batch_size=1):
        img = img.resize((width, height), resample=Image.BICUBIC)
        img = np.array(img).astype(np.float32) / 255.0
        img = np.vstack([img[None].transpose(0, 3, 1, 2)] * batch_size)
        img = torch.from_numpy(img)
        return img

    zeros = np.zeros_like(np.array(image))
    dice_np = [np.array(image)] + [zeros] * 5

    output_image = py360.c2e(dice_np, height, width, cube_format='list')
    control_image = Image.fromarray(output_image.astype(np.uint8))
    control_image = to_tensor(control_image)
    backgr_image = to_tensor(image)
    mask_image = to_tensor(mask)

    control_image = (1 - mask_image) * backgr_image + mask_image * control_image
    control_image = torch.cat([mask_image[:, :1, :, :], control_image], dim=1)
    return control_image


def load_model(model_id: str = "sd15/diffusion360"):
    model_dir = shared.MVDIFF_LOCAL_MODELS[model_id]
    ctrl_path = model_dir + '/sd-i2p'
    model_path = model_dir + '/sr-base'

    ctrlnet = ControlNetModel.from_pretrained(ctrl_path, torch_dtype=shared.dtype)
    model = Image2PanoBaseline.from_pretrained(model_path, torch_dtype=shared.dtype,
                                        controlnet=ctrlnet).to(device=shared.device)
    # model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)

    if shared.low_vram:
        model = enable_lowvram_usage(model)
    return model


def inference(
    model, 

    positive_prompt: str,
    negative_prompt: str,

    image: Image.Image,
    mask: Image.Image | None = None,

    generator = None,
    image_size: int = 768,

    strength: float = 1.,
    diffusion_steps: int = 20, 
    guidance_scale: float = 9.5, 
    controlnet_scale: float = 1.,

    **kwargs
):
    height = image_size
    width = image_size * 2

    if mask is None:
        mask_path = Path(__file__).resolve().parent / 'diffusion360/img2pano/i2p-mask.jpg'
        mask_path = str(mask_path).replace('\\','/')
        mask = load_image(mask_path)

    image = image.resize((width, width))
    mask = mask.resize((width, height))
    control_image = process_control_image(image, mask, width=width, height=height)

    if '<360panorama>' not in positive_prompt:
        positive_prompt = f'<360panorama>, {positive_prompt}'

    gen_params = dict(
         prompt = positive_prompt,
negative_prompt = negative_prompt,
        width = width, 
        height = height, 
        strength = strength,
        image = control_image[:, 1:, :, :] / 0.5 - 1.0,
control_image = control_image, 
controlnet_conditioning_scale = controlnet_scale,
        guidance_scale = guidance_scale,
        num_inference_steps = diffusion_steps,
        generator = generator,
    )

    output = model(**gen_params).images[0]
    return output


if __name__ == "__main__":

    # Inputs
    prompt = "((a backdrop of a launching event)) at an extended exhibition hall, "\
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
        guidance_scale=11.27,
        generator=torch.manual_seed(1995),
    )
    output.save(f'./temp/{output_name}_x1.png')

