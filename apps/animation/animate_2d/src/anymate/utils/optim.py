import cv2
import numpy as np

from PIL import Image
from PIL.Image import Image as ImageClass
from pathlib import Path

from typing import List, Tuple
from torch import nn
import torch

from diffusers.hooks import apply_group_offloading
from diffusers.utils.import_utils import is_xformers_available

from ... import shared


def find_submodules(pipe):
    all_submodules = [
        m for m in [
            "text_encoder", "text_encoder_1", "text_encoder_2",
            "image_encoder", 
            "image_embedder",
            "unet", "transformer", "brushnet",
            "vae", "vae_1_0",
        ] if hasattr(pipe, m)
    ]
    return all_submodules


def enable_lowvram_usage(pipe, offload_only: bool = False):
    """
    Memory Optimization:
        https://huggingface.co/docs/diffusers/en/optimization/memory
    
    Group Offloading: 
        https://github.com/huggingface/diffusers/pull/10503
    """
    if not offload_only:
        # Slicing
        pipe.enable_vae_tiling()
        pipe.enable_vae_slicing()
        pipe.enable_attention_slicing()
        
        if is_xformers_available():
            pipe.enable_xformers_memory_efficient_attention()

    # Offloading
    # pipe.enable_model_cpu_offload()

    modules = find_submodules(pipe)
    for m in modules:
        module = getattr(pipe, m)
        if not isinstance(module, nn.Module):
            continue

        offload_config = dict(
            non_blocking = True,
            onload_device = shared.device, 
            offload_device = torch.device('cpu'), 
        )

        if m.startswith(('text_encoder','image_encoder','image_embedder')):
            offload_config.update(dict(offload_type="block_level", num_blocks_per_group=2))

        elif m.startswith(('unet','transformer')):
            offload_config.update(dict(offload_type="leaf_level", use_stream=True, low_cpu_mem_usage=True))

        elif m.startswith(('vae','brushnet')):
            offload_config.update(dict(offload_type="leaf_level"))

        apply_group_offloading(module, **offload_config)

    return pipe


def blend_images(
        image_main: ImageClass, 
        image_aux: ImageClass,
        image_mask: ImageClass,
        blur_size: int = 19,
        blur_sigma: float = 0,
    ):

    image_main = np.array(image_main)
    image_aux  = np.array(image_aux)
    image_mask = np.array(image_mask)

    assert image_main.shape[:2] == image_aux.shape[:2] == image_mask.shape[:2]

    # Normalize mask to [0, 1] if it's not already
    if image_mask.dtype != np.float32:
        image_mask = image_mask.astype(np.float32) / 255.0

    # If mask is 1-channel, make it 3-channel if images are color
    if len(image_mask.shape) == 2 \
    and len(image_main.shape) == 3:
        image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2BGR)

    # Apply Gaussian Blur to smooth the mask
    blur_kernel = (blur_size, blur_size)
    blur_mask = cv2.GaussianBlur(image_mask, blur_kernel, blur_sigma)

    # Blend images
    image_blended = image_main * (1 - blur_mask) + image_aux * blur_mask
    image_blended = np.clip(image_blended, 0, 255).astype(np.uint8)
    image_blended = Image.fromarray(image_blended)

    return image_blended

