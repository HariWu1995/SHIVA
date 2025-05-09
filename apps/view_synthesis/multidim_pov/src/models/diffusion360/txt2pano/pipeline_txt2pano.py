# Copyright © Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import random
import numpy as np

import torch
from diffusers import (
    ControlNetModel, 
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler as   BaseScheduler,
            UniPCMultistepScheduler as RefineScheduler,
)
from PIL import Image

from .pipeline_base   import StableDiffusionBlendExtendPipeline           as Txt2PanoBaseline
from .pipeline_refine import StableDiffusionControlNetImg2ImgPanoPipeline as Txt2PanoRefiner

from ...utils import enable_lowvram_usage
from .... import shared


class StableDiffusionText2PanoramaPipeline(DiffusionPipeline):
    """ 
    Stable Diffusion for 360 Panorama Image Generation Pipeline.
    
    Example:
    >>> import torch
    >>> from txt2pano import StableDiffusionText2PanoramaPipeline
    >>> prompt = 'The mountains'
    >>> input = {'prompt': prompt, 'upscale': True}
    >>> model_id = 'models/'
    >>> pipe = StableDiffusionText2PanoramaPipeline(model_id, torch_dtype=torch.float16)
    >>> output = pipe(input)
    >>> output.save('result.png')
    """
    def __init__(
        self, 
        model: str, 
        refine: bool = False, 
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
        model_base_id = model + '/sd-base'

        pipe = Txt2PanoBaseline.from_pretrained(model_base_id, torch_dtype=torch_dtype).to(device)
        pipe.scheduler = BaseScheduler.from_config(pipe.scheduler.config)

        if shared.low_vram:
            pipe = enable_lowvram_usage(pipe)
        self.pipe = pipe

        # init refine model
        if refine:
            ctrl_refine_id = model + '/sr-control'
            model_refine_id = model + '/sr-base'

            recontrolnet = ControlNetModel.from_pretrained(ctrl_refine_id, torch_dtype=torch_dtype)
            pipe_sr = Txt2PanoRefiner.from_pretrained(model_refine_id, torch_dtype=torch_dtype, controlnet=recontrolnet).to(device)
            # pipe_sr.scheduler = RefineScheduler.from_config(pipe.scheduler.config)

            if shared.low_vram:
                pipe_sr = enable_lowvram_usage(pipe_sr)
            self.pipe_sr = pipe_sr
        else:
            self.pipe_sr = None

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

        # Generation params
        guidance_scale: float = 7.0,
        num_inference_steps: int = 25,

        # Others
        height: int = 768,
        width: int = 1536,
        generator = None,
    ):
        output = self.pipe(
            f'<360panorama>, {positive_prompt}',
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=generator
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
            guidance_scale=refine_scale,
            strength=refine_strength,
            generator=generator,
        ).images[0]
        return output

    def __call__(
        self, 

        # Inputs
        prompt: str,
        positive_prompt: str = 'photorealistic, trend on artstation, ((best quality)), ((ultra high res))',
        negative_prompt: str = 'worst quality, low quality, logo, text, watermark, monochrome, blur, '\
                               'persons, complex texture, small objects, sheltered, complex lighting',
        # Base params
        guidance_scale: float = 7.0,
        num_inference_steps: int = 20,
        
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
            prompt, negative_prompt,
            guidance_scale, num_inference_steps, 
            height, width, generator,
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

