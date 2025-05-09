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

from .. import shared


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


# OpenCV version
def _resize_and_center_crop(img: np.ndarray, size: int) -> np.ndarray:
    H, W, _ = img.shape
    if H == W:
        img = cv2.resize(img, (size, size))
    elif H > W:
        current_size = int(size * H / W)
        img = cv2.resize(img, (size, current_size))
        margin_l = (current_size - size) // 2
        margin_r =  current_size - size - margin_l
        img = img[margin_l:-margin_r, :]
    else:
        current_size = int(size * W / H)
        img = cv2.resize(img, (current_size, size))
        margin_t = (current_size - size) // 2
        margin_b =  current_size - size - margin_t
        img = img[:, margin_t:-margin_b]
    return img


def resize_and_center_crop(img: ImageClass, size: int) -> ImageClass:
    W, H = img.size  # PIL uses (width, height)
    if H == W:
        img = img.resize((size, size), Image.LANCZOS)
    elif H > W:
        new_height = int(size * H / W)
        img = img.resize((size, new_height), Image.LANCZOS)
        margin_top = (new_height - size) // 2
        img = img.crop((0, margin_top, size, margin_top + size))
    else:
        new_width = int(size * W / H)
        img = img.resize((new_width, size), Image.LANCZOS)
        margin_left = (new_width - size) // 2
        img = img.crop((margin_left, 0, margin_left + size, size))
    return img

