import os
from tqdm import tqdm
from PIL import Image

import numpy as np
import cv2

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from ..models.pano.MVGenModel import MultiViewBaseModel


class PanoOutpaintor(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()

        self.lr         = config['train']['lr']
        self.max_epochs = config['train']['max_epochs'] if 'max_epochs' in config['train'] else 0
        
        self.diff_timestep  = kwargs.get( 'diff_timestep', config['model']['diff_timestep'])
        self.guidance_scale = kwargs.get('guidance_scale', config['model']['guidance_scale'])

        self.tokenizer    = CLIPTokenizer.from_pretrained(config['model']['model_id'], subfolder="tokenizer", torch_dtype=torch.float16)
        self.text_encoder = CLIPTextModel.from_pretrained(config['model']['model_id'], subfolder="text_encoder", torch_dtype=torch.float16)

        self.vae, self.scheduler, unet = self.load_model(config['model']['model_id'])
        self.mv_base_model = MultiViewBaseModel(unet, config['model'])
       
    def load_model(self, model_id):
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        vae.eval()
        return vae, scheduler, unet

    @torch.no_grad()
    def encode_text(self, text, device):
        text_inputs = self.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length,
            truncation=True, 
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        if hasattr(self.text_encoder.config, "use_attention_mask") \
                and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
        return prompt_embeds[0].float(), prompt_embeds[1]

    @torch.no_grad()
    def encode_image(self, x_input, vae):
        b = x_input.shape[0]

        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)
        z = z.sample()

        # use the scaling factor from the vae config
        z = z * vae.config.scaling_factor
        z = z.float()
        return z

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b, m = latents.shape[0:2]
        latents = (1 / vae.config.scaling_factor * latents)

        images = []
        for j in tqdm(range(m)):
            image = vae.decode(latents[:, j]).sample
            images.append(image)
        image = torch.stack(images, dim=1)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 1, 3, 4, 2).float().numpy()
        image = (image * 255).round().astype('uint8')

        return image

    def prepare_mask_latents(self, mask, masked_image, batch_size, height, width):
        mask = F.interpolate(mask, size=(height // 8, width // 8))
        masked_image_latents = self.encode_image(masked_image, self.vae)
        return mask, masked_image_latents

    def gen_cls_free_guide_pair(self, latents, timestep, prompt_embd, batch):
        latents  = torch.cat([latents] * 2)
        timestep = torch.cat([timestep] * 2)
        
        R = torch.cat([batch['R']] * 2)
        K = torch.cat([batch['K']] * 2)
      
        meta = {
            'K': K,
            'R': R,
        }
        return latents, timestep, prompt_embd, meta

    @torch.no_grad()
    def forward_cls_free(self, latent_mm, _timestep, prompt_embd, batch, model):
        latents, _timestep, _prompt_embd, meta = self.gen_cls_free_guide_pair(
            latent_mm, _timestep, prompt_embd, batch
        )

        noise_pred = model(latents, _timestep, _prompt_embd, meta)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)
        return noise_pred
    
    def prepare_mask_image(self, images):
        bs, m, _, h, w = images.shape
        
        mask = torch.ones(bs, m, 1, h, w, device=images.device)
        mask[:, 0] = 0
        masked_image = images * (mask < 0.5)
        
        mask_latents = []
        image_latents = []
        
        for i in range(m):
            _mask, \
            _image_latent = self.prepare_mask_latents(mask[:, i],
                                              masked_image[:, i], bs, h, w)
            mask_latents.append(_mask)
            image_latents.append(_image_latent)

        mask_latents  = torch.stack( mask_latents, dim=1)
        image_latents = torch.stack(image_latents, dim=1)
        return mask_latents, image_latents

    @torch.no_grad()
    def inference(self, batch):
        images = batch['images']

        device = images.device
        dtype = images.dtype

        bs, m, h, w, _ = images.shape
        images = rearrange(images, 'bs m h w c -> bs m c h w')

        print("\nEncoding image ...")
        mask_latents, \
        masked_image_latents = self.prepare_mask_image(images)

        latents = torch.randn(bs, m, 4, h//8, w//8, device=device)

        print("\nEncoding text ...")
        prompt_embds = []
        for prompt in batch['prompt']:
            prompt_embds.append(self.encode_text(prompt, device)[0])
        prompt_embds = torch.stack(prompt_embds, dim=1)
        
        prompt_null = self.encode_text('', device)[0]
        prompt_embd = torch.cat([prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])

        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps
        
        print("\nDiffusing ...")
        for i, t in tqdm(enumerate(timesteps), total=self.diff_timestep):
            _timestep = torch.cat([t[None, None]] * m, dim=1)
            latent_mm = torch.cat([latents, mask_latents, masked_image_latents], dim=2)

            noise_pred = self.forward_cls_free(latent_mm, _timestep, prompt_embd, batch, self.mv_base_model)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        print("\nDecoding ...")
        images_pred = self.decode_latent(latents, self.vae)
        return images_pred

