# Copyright © Alibaba, Inc. and its affiliates.
import os

from pathlib import Path
from typing import Any, Dict

import random
import numpy as np

import torch
from diffusers.utils import load_image
from diffusers import (
    ControlNetModel, 
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler as   BaseScheduler,
            UniPCMultistepScheduler as RefineScheduler,
)
from PIL import Image
import py360convert as py360

from .pipeline_base   import StableDiffusionImage2PanoPipeline         as Img2PanoBaseline
from .pipeline_refine import StableDiffusionControlNetImg2PanoPipeline as Img2PanoRefiner

from ...utils import enable_lowvram_usage
from .... import shared


class StableDiffusionImage2PanoramaPipeline(DiffusionPipeline):
    """ 
    Stable Diffusion for 360 Panorama Image Generation Pipeline.
    
    Example:
    >>> import torch
    >>> from img2pano import StableDiffusionImage2PanoramaPipeline
    >>> image = load_image("./data/i2p-image.jpg").resize((512, 512))
    >>> mask = load_image("./data/i2p-mask.jpg")
    >>> prompt = 'The office room'
    >>> input = {'prompt': prompt, 'image': image, 'mask': mask, 'upscale': False}
    >>> model_id = 'models'
    >>> pipe = StableDiffusionImage2PanoramaPipeline(model_id, torch_dtype=torch.float16)
    >>> output = pipe(input)
    >>> output.save('result.png')
    """
    def __init__(
        self, 
        model: str, 
        refine: bool = False, 
        upscale: bool = False,
        **kwargs
    ):
        """
        Use `model` to create a stable diffusion pipeline for 360 panorama image generation.
        Args:
            model: model_id on modelscope hub / model_dir in local.
            device: str = 'cuda'
        """
        super().__init__()

        device = shared.device
        torch_dtype = shared.dtype

        # init base model
        ctrl_base_id = model + '/sd-i2p'
        model_base_id = model + '/sr-base'

        controlnet = ControlNetModel.from_pretrained(ctrl_base_id, torch_dtype=torch_dtype)
        pipe = Img2PanoBaseline.from_pretrained(model_base_id, torch_dtype=torch_dtype, controlnet=controlnet).to(device)
        # pipe.scheduler = BaseScheduler.from_config(pipe.scheduler.config)

        if shared.low_vram:
            pipe = enable_lowvram_usage(pipe)
        self.pipe = pipe

        # init refine model
        if refine:
            ctrl_refine_id = model + '/sr-control'
            model_refine_id = model + '/sr-base'

            recontrolnet = ControlNetModel.from_pretrained(ctrl_refine_id, torch_dtype=torch_dtype)
            pipe_sr = Img2PanoRefiner.from_pretrained(model_refine_id, torch_dtype=torch_dtype, controlnet=recontrolnet).to(device)
            # pipe_sr.scheduler = RefineScheduler.from_config(pipe_sr.scheduler.config)

            if shared.low_vram:
                pipe_sr = enable_lowvram_usage(pipe_sr)
            self.pipe_sr = pipe_sr
        else:
            self.pipe_sr = None

        # init upscale model
        if upscale:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer as RealESRGAN

            model_path = os.path.dirname(model) + '/RealESRGAN_x2plus.pth'
            model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                    num_block=23, num_grow_ch=32, scale=2)
            self.upsampler = RealESRGAN(
                model=model_arch, model_path=model_path, dni_weight=None,
                scale=2, tile=384, tile_pad=20, pre_pad=20, half=False, device=device,
            )
        else:
            self.upsampler = None

    @staticmethod
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

    @staticmethod
    def blend_h(a, b, blend_extent):
        blend_extent = min(a.shape[1], b.shape[1], blend_extent)
        for x in range(blend_extent):
            b[:, x, :] = \
            b[:, x, :]                 * (    x / blend_extent) + \
            a[:, -blend_extent + x, :] * (1 - x / blend_extent)
        return b

    def generate(
        self,

        # Inputs
        positive_prompt: str,
        negative_prompt: str,

        image: Image.Image,
        mask: Image.Image | None = None,

        # Generation params
        base_strength: float = 1.0,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 25,
        control_condition_scale: float = 1.0,

        # Others
        height: int = 768,
        width: int = 1536,
        generator = None,
    ):
        if mask is None:
            mask_path = Path(__file__).resolve().parent / 'i2p-mask.jpg'
            mask_path = str(mask_path).replace('\\','/')
            mask = load_image(mask_path)
        control_image = self.process_control_image(image, mask, width, height)

        output = self.pipe(
            f'<360panorama>, {positive_prompt}',
            negative_prompt=negative_prompt,
            image=(control_image[:, 1:, :, :] / 0.5 - 1.0),
    control_image=control_image,
    controlnet_conditioning_scale=control_condition_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=base_strength,
            height=height,
            width=width,
            generator=generator,
        ).images[0]
        return output

    def refine(
        self,

        # Inputs
        positive_prompt: str,
        negative_prompt: str,
        control_image: Image.Image,

        # Generation params
        refine_strength: float = 0.5,
        refine_scale: float = 7.,
        num_refinement_steps: int = 10,
        control_refine_scale: float = 1.0,

        # Others
        generator = None,
    ):
        output = self.pipe_sr(
             prompt=positive_prompt,
    negative_prompt=negative_prompt,
            image=control_image,
    control_image=control_image,
    controlnet_conditioning_scale=control_refine_scale,
            num_inference_steps=num_refinement_steps,
            strength=refine_strength,
      guidance_scale=refine_scale,
            generator=generator,
        ).images[0]
        return output

    def __call__(
        self, 

        # Inputs
        prompt: str,
        image: Image.Image,
        mask: Image.Image | None = None,

        positive_prompt: str = 'photorealistic, trend on artstation, ((best quality)), ((high res))',
        negative_prompt: str = 'worst quality, low quality, logo, text, watermark, monochrome, blur, '\
                               'persons, complex texture, small objects, sheltered, complex lighting',

        # Base params
        base_strength: float = 1.0,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 25,
        control_condition_scale: float = 1.0,
        
        # Refine params
        refine_strength: float = 0.5,
        refine_scale_x2: float = 7.,
        refine_scale_x4: float = 7.,
        num_refinement_steps: int = 15,
        control_refine_scale: float = 1.0,

        # Others
        height: int = 768,
        width: int = 1536,
        seed: int = -1,
    ) -> Dict[str, Any]:

        if seed == -1:
            seed = random.randint(0, 65535)
        generator = torch.manual_seed(seed)

        if isinstance(positive_prompt, str) \
              and len(positive_prompt) > 5:
            prompt = prompt + ', ' + positive_prompt

        ########################
        #       Baseline       #
        ########################

        print('\n\n Baselining ...')
        output = self.generate(
            prompt, negative_prompt, image, mask,
            base_strength, guidance_scale, num_inference_steps, 
            control_condition_scale, height, width, generator,
        )

        if not self.pipe_sr:
            return output

        #########################
        #       Refinement      #
        #########################

        print('\n\n Refining (upscale x2) ...')

        prompt = prompt.replace('<360panorama>', '')
        image = output.resize((width * 2, height * 2))

        output = self.refine(
            prompt, negative_prompt, image,
            refine_strength, refine_scale_x2, 
            num_refinement_steps,
            control_refine_scale, generator,
        )

        if not self.upsampler:
            return output

        #########################
        #       Upsampling      #
        #########################

        print('\n\n Upsampling (x2 -> x4) ...')

        output = output.resize((width * 2, height * 2))
        output = self.upsample(output)

        print('\n\n Refining (upscale x4) ...')

        image = output.resize((width * 4, height * 4))
        output = self.refine(
            prompt, negative_prompt, image,
            refine_strength, refine_scale_x4, 
            num_refinement_steps,
            control_refine_scale, generator,
        )

        return output

    def upsample(self, image: Image.Image):
        w, h, *_ = image.size

        blend_extend = 10
        outscale = 2

        image = np.array(image)
        image = np.concatenate([image, 
                                image[:, :blend_extend, :]], axis=1)
        image, _ = self.upsampler.enhance(image, outscale=outscale)
        image = self.blend_h(image, image, blend_extend*outscale)
        image = Image.fromarray(image[:, :w * outscale, :])
        return image


