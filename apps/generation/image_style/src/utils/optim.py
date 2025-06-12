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

