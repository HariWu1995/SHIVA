import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from pathlib import Path

from .lora import LORA_LONGLIST
from .model import IMAGENE_REMOTE_MODELS
from .ctrlnet import IMAGENE_CTRLNETS_FULL, IMAGENE_CTRLNETS_HALF
from .adapter import IMAGENE_ADAPTERS_FULL, IMAGENE_ADAPTERS_HALF, IP_ADAPTERS as REMOTE_IP_ADAPTERS
from .encoder import IMAGE_ENCODERS as REMOTE_IMAGE_ENCODERS, \
                      TEXT_ENCODERS as REMOTE_TEXT_ENCODERS

from ..utils import get_device_memory


MODEL_EXTENSIONS = ('.safetensors','.ckpt','.pt','.pth','.bin')


# FIXME: Hardcode to connect to multiple checkpoint directories
os.environ['SHIVA_CKPT_SD15']  = "D:/stable-diffusion/sd-15"
os.environ['SHIVA_CKPT_SDXL']  = "D:/stable-diffusion/sd-xl"
os.environ['SHIVA_CKPT_FLUX']  = "D:/stable-diffusion/Flux"
os.environ['SHIVA_CKPT_BRUSH'] = "D:/stable-diffusion/BrushNet"
os.environ['SHIVA_CKPT_IMGCODER'] = "E:/MMM"
os.environ['SHIVA_CKPT_TXTCODER'] = "E:/LLM"


CHECKPOINT_BRANCHES = dict()

for branch in ['sd15', 'sdxl', 'brush', 'flux', 'imgcoder', 'txtcoder']:
    CHECKPOINT_BRANCHES[branch] = os.environ.get(f'SHIVA_CKPT_{branch.upper()}', None)
    if CHECKPOINT_BRANCHES[branch] is not None:
        CHECKPOINT_BRANCHES[branch] = Path(CHECKPOINT_BRANCHES[branch])
    else:
        CHECKPOINT_BRANCHES[branch] = Path(__file__).resolve().parents[4] / f'checkpoints/{branch}'    
    if CHECKPOINT_BRANCHES[branch].exists():
        CHECKPOINT_BRANCHES[branch].mkdir(parents=True, exist_ok=True)


# Check if model exists
IMAGENE_LOCAL_MODELS = dict()

for m, path in IMAGENE_REMOTE_MODELS.items():
    model_version, model_name = m.split('/')

    if model_version in ['flux', 'brush']:
        model_path = CHECKPOINT_BRANCHES[model_version] / model_name
        if model_path.exists():
            IMAGENE_LOCAL_MODELS[m] = model_path
        continue

    elif model_version not in ['sd15', 'sdxl']:
        continue

    # Handle SD-15/XL
    model_hub = (
            "huggingface" if  "huggingface.co"  in path else (
                "civitai" if      "civitai.com" in path else (
                 "gdrive" if "drive.google.com" in path else "unknown"
            )
        )
    )

    if model_hub == 'unknown':
        continue
    elif model_hub in ["gdrive", "civitai"]:
        model_path = CHECKPOINT_BRANCHES[model_version] / "checkpoints" / f"{model_name}.safetensors"
    else:
        model_ext = Path(path).suffix
        if model_ext == '':
            model_path = CHECKPOINT_BRANCHES[model_version] / "checkpoints" / model_name
        elif model_ext in ['.bin', '.safetensors']:
            model_path = CHECKPOINT_BRANCHES[model_version] / "checkpoints" / f"{model_name}{model_ext}"
        else:
            continue
    
    if model_path.exists():
        IMAGENE_LOCAL_MODELS[m] = model_path


# Check if LoRA exists
set_name = lambda row: ' - '.join([str(x) for x in row if x not in ['', None, np.nan]])
set_path = lambda n, v: str(CHECKPOINT_BRANCHES[v] / "Lora" / f"{n}.safetensors")
is_exist = lambda x: os.path.isfile(x)

all_loras = []
namecols = ['category','article','version']
for v in ['sd15', 'sdxl']:
    LORA_LONGLIST[v]['name']  = LORA_LONGLIST[v][namecols].apply(set_name, axis=1)
    LORA_LONGLIST[v]['path']  = LORA_LONGLIST[v]['name'].apply(lambda x: set_path(x, v))
    LORA_LONGLIST[v]['exist'] = LORA_LONGLIST[v]['path'].apply(lambda p: is_exist(p))

    df = LORA_LONGLIST[v][LORA_LONGLIST[v]['exist']]
    df['sd_version'] = [v] * len(df)
    all_loras.append(df)

IMAGENE_LOCAL_LORAS = pd.concat(all_loras, axis=0, ignore_index=True)

# Check if Controlnet / T2I-Adapter exists
device_vram_Gb = get_device_memory(device_id=0, device_type='cuda') / 1024

if device_vram_Gb < 7:
    IMAGENE_REMOTE_CTRLNETS = IMAGENE_CTRLNETS_FULL
    IMAGENE_REMOTE_CTRLNETS.update(IMAGENE_CTRLNETS_HALF)

    IMAGENE_REMOTE_ADAPTERS = IMAGENE_ADAPTERS_FULL
    IMAGENE_REMOTE_ADAPTERS.update(IMAGENE_ADAPTERS_HALF)

else:
    IMAGENE_REMOTE_CTRLNETS = IMAGENE_CTRLNETS_HALF
    IMAGENE_REMOTE_CTRLNETS.update(IMAGENE_CTRLNETS_FULL)

    IMAGENE_REMOTE_ADAPTERS = IMAGENE_ADAPTERS_HALF
    IMAGENE_REMOTE_ADAPTERS.update(IMAGENE_ADAPTERS_FULL)

IMAGENE_LOCAL_CTRLNETS = dict()

for m, path in IMAGENE_REMOTE_CTRLNETS.items():

    model_version, ctrl_type, ctrl_name = m.split('/')

    model_ext = Path(path).suffix
    if model_ext not in ['.bin', '.safetensors']:
        continue

    model_path = CHECKPOINT_BRANCHES[model_version] / f"controlnet/control_{model_version}_{ctrl_name}{model_ext}"
    if model_path.exists():
        IMAGENE_LOCAL_CTRLNETS[m] = model_path

IMAGENE_LOCAL_ADAPTERS = dict()

for m, path in IMAGENE_REMOTE_ADAPTERS.items():

    model_version, ctrl_type, ctrl_name = m.split('/')

    model_ext = Path(path).suffix
    if model_ext not in ['.bin', '.safetensors']:
        continue

    model_path = CHECKPOINT_BRANCHES[model_version] / f"controlnet/adapter_{model_version}_{ctrl_name}{model_ext}"
    if model_path.exists():
        IMAGENE_LOCAL_ADAPTERS[m] = model_path

LOCAL_IP_ADAPTERS = dict()

for m, path in REMOTE_IP_ADAPTERS.items():

    model_version, ip_type, ip_name = m.split('/')

    model_ext = Path(path).suffix
    if model_ext not in ['.bin', '.safetensors']:
        continue

    model_path = CHECKPOINT_BRANCHES[model_version] / f"controlnet/{ip_name}{model_ext}"
    if model_path.exists():
        LOCAL_IP_ADAPTERS[m] = model_path


# Check if text-/image-encoder exists
LOCAL_IMAGE_ENCODERS = dict()

for m, path in REMOTE_IMAGE_ENCODERS.items():

    model_path = CHECKPOINT_BRANCHES['imgcoder'] / m
    if model_path.exists():
        LOCAL_IMAGE_ENCODERS[m] = model_path

LOCAL_TEXT_ENCODERS = dict()

for m, path in REMOTE_TEXT_ENCODERS.items():

    model_path = CHECKPOINT_BRANCHES['txtcoder'] / m
    if model_path.exists():
        LOCAL_TEXT_ENCODERS[m] = model_path


if __name__ == "__main__":
    import json
    # print(json.dumps({k: str(v) for k, v in CHECKPOINT_BRANCHES.items()}, indent=4))
    # print(json.dumps({k: str(v) for k, v in IMAGENE_LOCAL_MODELS.items()}, indent=4))
    # print(json.dumps({k: str(v) for k, v in IMAGENE_LOCAL_CTRLNETS.items()}, indent=4))
    # print(json.dumps({k: str(v) for k, v in IMAGENE_LOCAL_ADAPTERS.items()}, indent=4))

    # print(LORA_LONGLIST['sd15'][['category','article','version','name']])
    # print(LORA_LONGLIST['sdxl'][['category','article','version','name']])
    print(IMAGENE_LOCAL_LORAS[['category','article','version','name']])
    
