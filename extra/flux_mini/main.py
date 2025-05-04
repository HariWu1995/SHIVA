from dataclasses import dataclass
from typing import Union, Optional, List, Any, Dict

import os
import gradio as gr
import numpy as np
import random
import torch

from diffusers import  FluxPipeline, AutoencoderTiny, AutoencoderKL, FlowMatchEulerDiscreteScheduler as Scheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from safetensors.torch import load_file

from .model import FluxParams, Flux as FluxModel
from .default import mini_params, MAX_IMAGE_SIZE, MAX_SEED


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@torch.inference_mode()
def flux_pipe_call_that_returns_an_iterable_of_images(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    max_sequence_length: int = 512,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    timesteps: List[int] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    vae: Optional[Any] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = self._execution_device

    # 3. Encode prompt
    lora_scale = joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None
    prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    self._num_timesteps = len(timesteps)

    # Handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])
    else:
        guidance = None

    # 6. Denoising loop
    for i, t in enumerate(timesteps):
        if self.interrupt:
            continue

        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

        # Yield intermediate result
        latents_for_image = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents_for_image = (latents_for_image / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents_for_image, return_dict=False)[0]
        yield self.image_processor.postprocess(image, output_type=output_type)[0]
        
        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        torch.cuda.empty_cache()

    # Final image using vae
    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]

    self.maybe_free_model_hooks()
    torch.cuda.empty_cache()

    yield self.image_processor.postprocess(image, output_type=output_type)[0]


@dataclass
class ModelSpec:
    params: FluxParams
    repo_id: str 
    repo_flow: str 
    repo_ae: str 
    repo_id_ae: str


def load_pipeline(
    path_flux_mini: str,
    path_flux_dev: str | None = None,
    path_textcoder: str | None = None,  # use smaller text-encoder instead of T5-XXL
    dtype = torch.bfloat16,
    device = "cuda" if torch.cuda.is_available() else "cpu",
):
    if path_flux_dev is None:
        path_flux_dev = path_flux_mini.replace('mini', 'dev')

    config = ModelSpec(
        repo_id=path_flux_mini, #"TencentARC/flux-mini",
        repo_flow="flux-mini.safetensors",
        repo_id_ae=path_flux_dev, #"black-forest-labs/FLUX.1-dev",
        repo_ae="ae.safetensors",
        params=FluxParams(**mini_params)
    )
    
    torch.cuda.empty_cache()

    # Load model
    scheduler = Scheduler.from_pretrained(path_flux_dev, subfolder="scheduler")
    encoder = AutoencoderKL.from_pretrained(path_flux_dev, subfolder="vae", torch_dtype=dtype).to(device)
    textcoder = CLIPTextModel.from_pretrained(path_flux_dev, subfolder="text_encoder").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(path_flux_dev, subfolder="tokenizer")

    if path_textcoder is not None and os.path.isdir(path_textcoder):
        textcoder_2 = T5EncoderModel.from_pretrained(path_textcoder).to(device)
        tokenizer_2 = T5TokenizerFast.from_pretrained(path_textcoder)
    else:
        textcoder_2 = T5EncoderModel.from_pretrained(path_flux_dev, subfolder="text_encoder_2").to(device)
        tokenizer_2 = T5TokenizerFast.from_pretrained(path_flux_dev, subfolder="tokenizer_2")

    transformer = FluxModel(config.params)
    model_weights = load_file(os.path.join(path_flux_mini, "flux-mini.safetensors"), device=str(device))
    missed, unexpected = transformer.load_state_dict(model_weights, strict=True)

    # Combine into pipeline
    pipe = FluxPipeline(scheduler, encoder, textcoder, tokenizer, textcoder_2, tokenizer_2, transformer)
    pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)
    return pipe


def infer(
    prompt, 
    seed=42, 
    randomize_seed=False, 
    width=1024, 
    height=1024, 
    guidance_scale=3.5, 
    num_inference_steps=28, 
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    
    for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
        output_type="pil",
        vae=vae,
    ):
        yield img, seed
    
