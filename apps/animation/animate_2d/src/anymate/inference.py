from PIL import Image

import math
import numpy as np

import torch
import torchvision.transforms as T

from einops import rearrange, repeat
from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTextModel, CLIPTokenizer

from .models.unet_3d_condition_mask import UNet3DConditionModel as UNet3DCond
from .models.pipeline import LatentToVideoPipeline
from .utils.common import tensor_to_vae_latent, DDPM_forward
from .utils.optim import enable_lowvram_usage
from .utils.io import save_video_frames

from ..path import ANIMATE_LOCAL_MODELS
from .. import shared


def load_pipeline(model_id: str = 'animateanything'):
    assert model_id.startswith('animateanything')
    model_path = ANIMATE_LOCAL_MODELS[model_id]

    # Load sub-modules
    unet = UNet3DCond.from_pretrained(model_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")

    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    textcoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

    # Load pipeline
    # pipeline = LatentToVideoPipeline.from_pretrained(model_path)
    pipeline = LatentToVideoPipeline(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=textcoder,
    )
    pipeline = pipeline.to(device=shared.device, dtype=shared.dtype)

    # Enable low VRAM strategy
    if shared.low_vram:
        pipeline = enable_lowvram_usage(pipeline, offload_only=True)
    return pipeline


def run_inference(
    pipeline,
    control_image: Image.Image or np.ndarray,
    control_mask: Image.Image or np.ndarray,
    positive_prompt: str = 'moving',
    negative_prompt: str = 'bad quality',
    target_size: tuple = (512, 512),
    num_frames: int = 16,
    sampling_steps: int = 25,
    guidance_scale: float = 9.0,
    motion_scale: float = 5.0,   # Larger value means larger motion but less identity consistency
    seed: int = -1,
): 
    if isinstance(seed, (int, float)):
        torch.manual_seed(int(seed))
    else:
        torch.seed()
    seed = torch.initial_seed()

    # Preprocess
    init_latents, timesteps, \
   image_latents, mask = preprocess(pipeline, control_image, control_mask,
                                    target_size, 8, num_frames, sampling_steps)

    motion_strength = motion_scale * mask.mean().item()

    # Generate
    with torch.no_grad():
        video_frames = pipeline(
            prompt = positive_prompt,
    negative_prompt = negative_prompt,
            latents = init_latents,
    condition_latent = image_latents,
            mask = mask,
            width = target_size[0],
            height = target_size[1],
            num_frames = num_frames,
    num_inference_steps = sampling_steps,
            guidance_scale = guidance_scale,
            motion = [motion_strength],
            timesteps = timesteps,
        ).frames

    return video_frames


def preprocess(
    pipeline,
    control_image: Image.Image or np.ndarray,
    control_mask: Image.Image or np.ndarray,
    target_size: tuple = (512, 512),
    block_size: int = 8,
    num_frames: int = 16,
    sampling_steps: int = 25,
):
    if isinstance(control_image, np.ndarray):
        control_image = Image.fromarray(control_image)
    control_image = control_image.convert('RGB')

    if isinstance(control_mask, Image.Image):
        control_mask = np.array(control_mask)
    assert len(control_mask.shape) in [2,3], \
        f"{control_mask.shape} is not supported shape for `control_mask`."
    if len(control_mask.shape) == 3:
        control_mask = control_mask[:,:,-1]

    # Scale
    width, height = control_image.size
    tgt_w, tgt_h = target_size

    scale = math.sqrt((width * height) / (tgt_w * tgt_h))

    width  = round( width / scale / block_size) * block_size
    height = round(height / scale / block_size) * block_size
    
    # Latents
    vae_processor =  VaeImageProcessor()
    input_image = vae_processor.preprocess(control_image, height, width)
    input_image = input_image.unsqueeze(0).to(device=pipeline.device, dtype=pipeline.dtype)
    image_latents = tensor_to_vae_latent(input_image, pipeline.vae)

    control_mask[control_mask != 0] = 255
    if control_mask.sum() == 0:
        control_mask[:] = 255
    # Image.fromarray(control_mask, mode='L').save('.temp/control_mask.png')

    # Diffusion
    init_latents, timesteps = DDPM_forward(image_latents, sampling_steps, num_frames, pipeline.scheduler) 
    b, c, f, h, w = init_latents.shape

    # Resize mask
    mask = T.ToTensor()(control_mask)
    mask = mask.to(device=pipeline.device, dtype=pipeline.dtype)
    mask = T.Resize([h, w], antialias=False)(mask)
    mask = rearrange(mask, 'b h w -> b 1 1 h w')

    return init_latents, timesteps, image_latents, mask


if __name__ == "__main__":

    pipeline = load_pipeline()

    # Sample
    image = Image.open('./temp/image_mask/pig0.jpg')
    mask = Image.open('./temp/image_mask/pig0_label.jpg')

    # Inference
    video_frames = run_inference(
        pipeline,
        control_image = image,
        control_mask = mask,
        positive_prompt = 'talking',
        negative_prompt = 'bad quality',
    )

    # Saving
    output_path = './temp/anymate.mp4'
    try:
        import imageio
        imageio.mimwrite(output_path, video_frames, fps=8)
    except Exception:
        save_video_frames(video_frames, output_path, revert_rgb=True, fps=5)
