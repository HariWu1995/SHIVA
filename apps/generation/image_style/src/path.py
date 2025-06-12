import os
from pathlib import Path


# FIXME: Hardcode to connect to multiple checkpoint directories
os.environ['SHIVA_CKPT_SDXL']  = "D:/stable-diffusion/sd-xl"
os.environ['SHIVA_CKPT_STYLE'] = "E:/stable-diffusion"
os.environ['SHIVA_CKPT_IMGCODER'] = "E:/MMM"


#############################################
#           (Base) Diffusion Models         #
#############################################

CHECKPOINT_BRANCHES = dict()

for branch in ['sdxl']:
    CHECKPOINT_BRANCHES[branch] = os.environ.get(f'SHIVA_CKPT_{branch.upper()}', None)
    if CHECKPOINT_BRANCHES[branch] is not None:
        CHECKPOINT_BRANCHES[branch] = Path(CHECKPOINT_BRANCHES[branch])
    else:
        CHECKPOINT_BRANCHES[branch] = Path(__file__).resolve().parents[4] / f'checkpoints/{branch}'    
    if CHECKPOINT_BRANCHES[branch].exists():
        CHECKPOINT_BRANCHES[branch].mkdir(parents=True, exist_ok=True)

SDIFF_REMOTE_MODELS = {
    "sdxl/sdxl_base_v1"             : "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
    "sdxl/sdxl_refine_v1"           : "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors",
    "sdxl/real_stock_photo_v2"      : "https://huggingface.co/alexgenovese/checkpoint/resolve/021e192bd744c48a85f8ae1832662e77beb9aac7/realisticStockPhoto_v20.safetensors",
    "sdxl/product_photo_midjourney" : "https://huggingface.co/alexgenovese/checkpoint/resolve/021e192bd744c48a85f8ae1832662e77beb9aac7/product-photography-midjourney.safetensors",

    "sdxl/juggernaut"               : "https://civitai.com/api/download/models/782002?type=Model&format=SafeTensor&size=full&fp=fp16",
    "sdxl/icbinp"                   : "https://civitai.com/api/download/models/399481?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "sdxl/dreamshaper_light"        : "https://civitai.com/api/download/models/354657?type=Model&format=SafeTensor&size=full&fp=fp16",
    "sdxl/epic_realism_v16"         : "https://civitai.com/api/download/models/1522905?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "sdxl/ahavietnam_realistic_v2"  : "https://civitai.com/api/download/models/137827?type=Model&format=SafeTensor&size=full&fp=fp16",
    "sdxl/sdvn_real_detail_face"    : "https://civitai.com/api/download/models/134461?type=Model&format=SafeTensor&size=full&fp=fp16",
}

SDIFF_LOCAL_MODELS = {
    "sdxl/sdxl_base_v1"             : str(CHECKPOINT_BRANCHES['sdxl'] / "checkpoints/sdxl_base_v1.safetensors"),
    "sdxl/sdxl_refine_v1"           : str(CHECKPOINT_BRANCHES['sdxl'] / "checkpoints/sdxl_refine_v1.safetensors"),
    "sdxl/juggernaut"               : str(CHECKPOINT_BRANCHES['sdxl'] / "checkpoints/juggernaut.safetensors"),
    "sdxl/icbinp"                   : str(CHECKPOINT_BRANCHES['sdxl'] / "checkpoints/icbinp.safetensors"),
    "sdxl/dreamshaper_light"        : str(CHECKPOINT_BRANCHES['sdxl'] / "checkpoints/dreamshaper_light.safetensors"),
    "sdxl/epic_realism_v16"         : str(CHECKPOINT_BRANCHES['sdxl'] / "checkpoints/epic_realism_v16.safetensors"),
    "sdxl/ahavietnam_realistic_v2"  : str(CHECKPOINT_BRANCHES['sdxl'] / "checkpoints/ahavietnam_realistic_v2.safetensors"),
    "sdxl/sdvn_real_detail_face"    : str(CHECKPOINT_BRANCHES['sdxl'] / "checkpoints/sdvn_real_detail_face.safetensors"),
    "sdxl/real_stock_photo_v2"      : str(CHECKPOINT_BRANCHES['sdxl'] / "checkpoints/real_stock_photo_v2.safetensors"),
    "sdxl/product_photo_midjourney" : str(CHECKPOINT_BRANCHES['sdxl'] / "checkpoints/product_photo_midjourney.safetensors"),
}

SDIFF_LOCAL_MODELS = {m: p.replace('\\', '/') for m, p in SDIFF_LOCAL_MODELS.items()}


##########################################
#           STYLIZATION MODELS           #
##########################################

CHECKPOINT_ROOT = os.environ.get('SHIVA_CKPT_STYLE', None)
if CHECKPOINT_ROOT is not None:
    STYLIZATION_DIR = Path(CHECKPOINT_ROOT)
else:
    STYLIZATION_DIR = Path(__file__).parents[4] / 'checkpoints/stylization'

if os.path.isdir(STYLIZATION_DIR) is False:
    os.makedirs(STYLIZATION_DIR)


STYLE_REMOTE_MODELS = {
    "sdxl/csgo"     : "https://huggingface.co/InstantX/CSGO/resolve/main/csgo.bin",
    "sdxl/csgo_4_32": "https://huggingface.co/InstantX/CSGO/resolve/main/csgo_4_32.bin",
    "sdxl/csgo_ctrl": "https://huggingface.co/TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic",
}


STYLE_LOCAL_MODELS = {
    "sdxl/csgo"     : str(STYLIZATION_DIR / "Instant-Style/csgo.bin"),
    "sdxl/csgo_4_32": str(STYLIZATION_DIR / "Instant-Style/csgo_4_32.bin"),
    "sdxl/csgo_ctrl": str(STYLIZATION_DIR / "Instant-Style/TTPLanet_Controlnet_Tile_Realistic"),
}

STYLE_LOCAL_MODELS = {m: p.replace('\\', '/') for m, p in STYLE_LOCAL_MODELS.items()}


#############################################
#           (Auxilliary) Encoders           #
#############################################

IMG_ENCODER_ROOT = os.environ.get('SHIVA_CKPT_IMGCODER', None)
if IMG_ENCODER_ROOT is not None:
    IMG_ENCODER_ROOT = Path(IMG_ENCODER_ROOT)
else:
    IMG_ENCODER_ROOT = Path(__file__).parents[4] / 'checkpoints/iencoder'

if os.path.isdir(IMG_ENCODER_ROOT) is False:
    os.makedirs(IMG_ENCODER_ROOT)

REMOTE_IMAGE_ENCODERS = {
    "clip-vit-h14-laion2B"  : "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin",
    "clip-vit-bigg14"       : "https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models/image_encoder",
}

LOCAL_IMAGE_ENCODERS = {
    "clip-vit-h14-laion2B"  : str(IMG_ENCODER_ROOT / "clip-vit-h14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"),
    "clip-vit-bigg14"       : str(IMG_ENCODER_ROOT / "clip-vit-bigg14"),
}

LOCAL_IMAGE_ENCODERS = {m: p.replace('\\', '/') for m, p in LOCAL_IMAGE_ENCODERS.items()}

