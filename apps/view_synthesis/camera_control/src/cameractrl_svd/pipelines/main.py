from pathlib import Path
from omegaconf import OmegaConf

import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection as CLIPVisionModelProj
from diffusers import EulerDiscreteScheduler, AutoencoderKLTemporalDecoder as TemporalDecoder
from diffusers.utils.import_utils import is_xformers_available

from ..models.pose_adaptor import CameraPoseEncoder
from ..models.unet import UNetSpatioTemporalConditionModelPoseCond as UNetTrajCond

from .optim import enable_lowvram_usage
from .pipeline_animation import StableVideoDiffusionPipelinePoseCond as SVDTrajPipeline

from ... import shared
from ...path import SDIFF_LOCAL_MODELS, CAMCTRL_LOCAL_MODELS


config_dir = str(Path(__file__).resolve().parents[1] / "configs/train_cameractrl")
config_dir = config_dir.replace('\\', '/')


def load_pipeline(model_name, device):
    assert model_name in ['svd', 'svdxt']

    if model_name == 'svd':
        model_path = SDIFF_LOCAL_MODELS['svd_im2vid']
        pose_path = CAMCTRL_LOCAL_MODELS['svd/camctrl']
        config_path = f"{config_dir}/svd_320_576_cameractrl.yaml"
    else:
        config_path = f"{config_dir}/svdxt_320_576_cameractrl.yaml"
        model_path = SDIFF_LOCAL_MODELS['svd_im2vid_xt']
        pose_path = CAMCTRL_LOCAL_MODELS['svd_xt/camctrl']

    # Load config
    model_config = OmegaConf.load(config_path)
    model_kwargs = dict(
        up_block_types   = model_config[  'up_block_types'],
        down_block_types = model_config['down_block_types'],
    )

    traj_kwargs = model_config['pose_encoder_kwargs']
    attn_kwargs = model_config['attention_processor_kwargs']

    # print(OmegaConf.to_yaml(traj_kwargs))
    # print(OmegaConf.to_yaml(attn_kwargs))

    # traj_kwargs = dict(**traj_kwargs)
    # attn_kwargs = dict(**attn_kwargs)

    # Load models
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(model_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelProj.from_pretrained(model_path, subfolder="image_encoder")
    
    vae = TemporalDecoder.from_pretrained(model_path, subfolder="vae")
    unet = UNetTrajCond.from_pretrained(model_path, subfolder="unet", **model_kwargs)
    
    vae.to(device)
    unet.to(device)
    image_encoder.to(device)

    # Load adaptor
    pose_encoder = CameraPoseEncoder(**traj_kwargs)

    attn_kwargs['enable_xformers'] = is_xformers_available()
    unet.set_pose_cond_attn_processor(**attn_kwargs)

    pose_adaptor_ckpt_dict = torch.load(pose_path, map_location=unet.device)
    pose_encoder_state_dict = pose_adaptor_ckpt_dict['pose_encoder_state_dict']

    pose_encoder_missed, \
    pose_encoder_unknown = pose_encoder.load_state_dict(pose_encoder_state_dict)
    assert len(pose_encoder_missed) == 0 
    assert len(pose_encoder_unknown) == 0

    attention_processor_state_dict = pose_adaptor_ckpt_dict['attention_processor_state_dict']
    _, attention_processor_unknown = unet.load_state_dict(attention_processor_state_dict, strict=False)
    assert len(attention_processor_unknown) == 0

    # Combine to pipeline
    pipeline = SVDTrajPipeline(
        vae=vae,
        unet=unet,
        scheduler=noise_scheduler,
        pose_encoder=pose_encoder,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
    )
    pipeline = pipeline.to(device)

    # Enable low VRAM strategy
    if shared.low_vram:
        pipeline = enable_lowvram_usage(pipeline, offload_only=True)
    return pipeline


