from omegaconf import OmegaConf
from safetensors import safe_open
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel as UNetCondition
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_vae_checkpoint, 
    convert_ldm_clip_checkpoint,
)

from ..utils import merge_lora2unet, convert_ldm_unet_checkpoint
from ..models.unet import UNet3DConditionModelPoseCond as UNet3DPoseCond
from ..models.pose_adaptor import CameraPoseEncoder
from .pipeline_animation import CameraCtrlPipeline


def load_pipeline(

    # Model checkpoints
    base_model_path: str, 
    animatediff_path: str,
    pose_adaptor_path: str, 

    # Model configs
    unet_extra_kwargs = dict(),
    pose_encoder_kwargs = dict(), 
    noise_scheduler_kwargs = dict(), 
    attention_processor_kwargs = dict(),

    # Customized models
    finetuned_ckpt: str = None, 
    image_lora_ckpt: str = None, image_lora_rank: int = 2, 
    unet_motion_ckpt: str = None, 

    # Misc.
    gpu_id: int = 0,
):

    unet2cond = UNetCondition.from_pretrained(base_model_path, subfolder="unet")
    vaencoder = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    textcoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder")

    # Load AnimateDiff-v3
    animate_dict = torch.load(animatediff_path, map_location='cpu')
    if 'state_dict' in animate_dict:
        animate_dict = animate_dict['state_dict']

    unet_animate = merge_lora2unet(unet2cond.state_dict(), animate_dict, lora_scale=1.0)
    del unet2cond, animate_dict
    
    # Load UNet3D
    unet = UNet3DPoseCond(**unet_extra_kwargs)
    unet.load_state_dict(unet_animate, strict=False)

    # Load Pose-Encoder
    posecoder = CameraPoseEncoder(**pose_encoder_kwargs)

    # Set Attention Processors
    unet.set_all_attn_processor(
        add_spatial_lora=image_lora_ckpt is not None,
        add_motion_lora=False,
        spatial_lora_kwargs={"lora_scale": 1.0, "lora_rank": image_lora_rank},
        motion_lora_kwargs={"lora_scale": 1.0, "lora_rank": -1},
        **attention_processor_kwargs
    )

    # Load LoRA (if any)
    if image_lora_ckpt is not None:
        print(f"\nLoading the LoRA checkpoint from {image_lora_ckpt}")
        lora_checkpoints = torch.load(image_lora_ckpt, map_location=unet.device)
        if 'lora_state_dict' in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints['lora_state_dict']
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0

    # Load LoRA (if any)
    if unet_motion_ckpt is not None:
        print(f"\nLoading the motion module checkpoint from {unet_motion_ckpt}")
        mm_checkpoints = torch.load(unet_motion_ckpt, map_location=unet.device)
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0

    # Load Pose-Adaptor
    pose_adaptor_checkpoint = torch.load(pose_adaptor_path, map_location='cpu')
    posecoder_state_dict = pose_adaptor_checkpoint['pose_encoder_state_dict']
    posecoder_missed, \
    posecoder_unknown = posecoder.load_state_dict(pose_encoder_state_dict)
    assert len(posecoder_unknown) == 0 \
        and len(posecoder_missed) == 0

    attention_processor_state_dict = pose_adaptor_checkpoint['attention_processor_state_dict']
    _, attn_proc_u = unet.load_state_dict(attention_processor_state_dict, strict=False)
    assert len(attn_proc_u) == 0

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    # Send to GPU
    vaencoder.to(gpu_id)
    textcoder.to(gpu_id)
    posecoder.to(gpu_id)
    unet.to(gpu_id)

    # Load pipeline
    pipe = CameraCtrlPipeline(
        unet=unet,
        vae=vaencoder,
        pose_encoder=posecoder,
        text_encoder=textcoder,
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
    )

    if finetuned_model is not None:
        load_finetuned_model(pipeline=pipe, finetuned_model=finetuned_model)

    # Enable low VRAM strategy
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    pipe = pipe.to(gpu_id)
    return pipe


def load_finetuned_model(pipeline, finetuned_model):
    print(f'Load base model from {finetuned_model}')
    if finetuned_model.endswith(".safetensors"):
        state_dict = {}
        with safe_open(finetuned_model, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    elif finetuned_model.endswith(".ckpt"):
        state_dict = torch.load(finetuned_model, map_location="cpu")

    # 1. vae
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, pipeline.vae.config)
    pipeline.vae.load_state_dict(converted_vae_checkpoint)
    
    # 2. unet
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(state_dict, pipeline.unet.config)
    _, unet = pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
    assert len(unet) == 0
    
    # 3. text_model
    pipeline.text_encoder = convert_ldm_clip_checkpoint(state_dict, text_encoder=pipeline.text_encoder)

    del state_dict
    return pipeline

