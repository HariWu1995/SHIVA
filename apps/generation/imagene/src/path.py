import os
from pathlib import Path

import torch
import numpy as np
import cv2

from .utils import get_device_memory


# FIXME: Hardcode to connect to multiple checkpoint directories
os.environ['SHIVA_CKPT_SD15']  = "D:/stable-diffusion/sd-15"
os.environ['SHIVA_CKPT_SDXL']  = "D:/stable-diffusion/sd-xl"
os.environ['SHIVA_CKPT_FLUX']  = "F:/__Checkpoints__/stable-diffusion/Flux"
os.environ['SHIVA_CKPT_BRUSH'] = "F:/__Checkpoints__/stable-diffusion/BrushNet"


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_BRANCHES = dict()

for branch in ['sd15', 'sdxl', 'brush', 'flux']:
    CHECKPOINT_BRANCHES[branch] = os.environ.get(f'SHIVA_CKPT_{branch.upper()}', None)
    if CHECKPOINT_BRANCHES[branch] is None:
        CHECKPOINT_BRANCHES[branch] = Path(__file__).parents[4] / f'checkpoints/{branch}'
    if CHECKPOINT_BRANCHES[branch].exists() is False:
        CHECKPOINT_BRANCHES[branch].mkdir(parents=True, exist_ok=True)


IMAGENE_MODELS = {

    # Flux Family (~24Gb, except Flux-mini ~6.4Gb after distillation)
    "flux1/mini" : "https://huggingface.co/TencentARC/flux-mini",
    "flux1/dev"  : "https://huggingface.co/black-forest-labs/FLUX.1-dev",
    "flux1/fill" : "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev",
    "flux1/redux": "https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev",

    # BrushNet (2.4 / 1.4 Gb but require SD backbone)
    "brushnet/random_mask"          : "https://drive.google.com/drive/folders/1hCYIjeRGx3Zk9WZtQf0s3nDGfeiwqTsN",
    "brushnet/random_mask_xl"       : "https://drive.google.com/drive/folders/1H2annwRr1HkUppbHe2gt9HHqO59EHXKc",
    "brushnet/segmentation_mask"    : "https://drive.google.com/drive/folders/1KPFFYblnovk4MU74OCBfS1EZU_jhBsse",
    "brushnet/segmentation_mask_xl" : "https://drive.google.com/drive/folders/1twv3gFka6RQ27tqHwVw0ocrv7qKk_q5r",

    # Stable Diffusion 1.5 (2-4Gb)
    "sd15/original_pruned_ema_only" : "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt",
    "sd15/inpaint_pruned_ema_only"  : "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt",

    "sd15/dreamshaper_v8_inpaint"  : "https://civitai.com/api/download/models/134084?type=Model&format=SafeTensor&size=full&fp=fp32",
    "sd15/dreamshaper_v8"          : "https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16",

    "sd15/abs_reality_v18_inpaint" : "https://civitai.com/api/download/models/131004?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "sd15/abs_reality_v18"         : "https://civitai.com/api/download/models/132760?type=Model&format=SafeTensor&size=pruned&fp=fp16",

    "sd15/real_vision_v51_inpaint" : "https://civitai.com/api/download/models/501286?type=Model&format=SafeTensor&size=full&fp=fp16",
    "sd15/real_vision_v51"         : "https://civitai.com/api/download/models/501240?type=Model&format=SafeTensor&size=full&fp=fp16",

    "sd15/epic_photogasm"          : "https://civitai.com/api/download/models/429454?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "sd15/epic_photogasm_inpaint"  : "https://civitai.com/api/download/models/201346?type=Model&format=SafeTensor&size=pruned&fp=fp16",

    "sd15/icbinp"          : "https://civitai.com/api/download/models/667760?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "sd15/icbinp_inpaint"  : "https://civitai.com/api/download/models/306943?type=Model&format=SafeTensor&size=pruned&fp=fp16",

    "sd15/anylora"          : "https://civitai.com/api/download/models/95489?type=Model&format=SafeTensor&size=full&fp=fp16",
    "sd15/anylora_inpaint"  : "https://civitai.com/api/download/models/131256?type=Model&format=SafeTensor&size=full&fp=fp16",

    "sd15/never_ending_dream"         : "https://civitai.com/api/download/models/64094?type=Model&format=SafeTensor&size=full&fp=fp16",
    "sd15/never_ending_dream_inpaint" : "https://civitai.com/api/download/models/74750?type=Model&format=SafeTensor&size=full&fp=fp16",

    "sd15/disney_pixar_cartoon" : "https://civitai.com/api/download/models/69832?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "sd15/xxmix_9realistic"     : "https://civitai.com/api/download/models/102222?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "sd15/perfect_deliberate"   : "https://civitai.com/api/download/models/253055?type=Model&format=SafeTensor&size=full&fp=fp32",

    # Stable Diffusion XL (6-7Gb)
    "sdxl/base_v1"   : "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
    "sdxl/refine_v1" : "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors",
    "sdxl/inpaint_v1": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1",

    "sdxl/icbinp" : "https://civitai.com/api/download/models/399481?type=Model&format=SafeTensor&size=pruned&fp=fp16",

    "sdxl/dreamshaper_light_inpaint" : "https://civitai.com/api/download/models/450187?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "sdxl/dreamshaper_light"         : "https://civitai.com/api/download/models/354657?type=Model&format=SafeTensor&size=full&fp=fp16",
    
    "sdxl/epic_realism_v16" : "https://civitai.com/api/download/models/1522905?type=Model&format=SafeTensor&size=pruned&fp=fp16",

    "sdxl/ahavietnam_realistic_v2" : "https://civitai.com/api/download/models/137827?type=Model&format=SafeTensor&size=full&fp=fp16",
    "sdxl/sdvn_real_detail_face"   : "https://civitai.com/api/download/models/134461?type=Model&format=SafeTensor&size=full&fp=fp16",
}


IMAGENE_LORAS = {}



device_vram_Gb = get_device_memory(device_id=0, device_type='cuda') / 1024

if device_vram_Gb < 24:
    del IMAGENE_MODELS["flux1/dev"]
    del IMAGENE_MODELS["flux1/fill"]
    del IMAGENE_MODELS["flux1/redux"]

if device_vram_Gb < 16:
    del MLM_LOCAL_MODELS["deepseek-7b-chat"]

if device_vram_Gb < 4:
    del MLM_LOCAL_MODELS["deepseek-r1-1b5-qwen"]

if device_vram_Gb < 3:
    del MLM_LOCAL_MODELS["llama-3.2-1b-instruct"]


LORA_LOCAL_MODELS = {}
LORA_REMOTE_MODELS = {}


def load_file_from_url(
    remote_url: str,
    model_dir: str | None = None,
    local_path: str | None = None,
    hash_prefix: str | None = None,
    progress: bool = True,
) -> str:
    raise NotImplementedError()


if os.environ.get('SHIVA_CKPT_PRELOAD', False):
    for model_name, model_path in MLM_LOCAL_MODELS.items():
        if os.path.isfile(model_path):
            continue
        load_file_from_url(remote_url=MLM_REMOTE_MODELS[model_name], 
                           local_path=MLM_LOCAL_MODELS[model_name])

