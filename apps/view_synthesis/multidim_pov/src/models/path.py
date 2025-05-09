import os
from pathlib import Path


# FIXME: Hardcode to connect to multiple checkpoint directories
os.environ['SHIVA_CKPT_SD20'] = "D:/stable-diffusion/sd-20"
os.environ['SHIVA_CKPT_SD21'] = "D:/stable-diffusion/sd-21"
os.environ['SHIVA_CKPT_MVDIFF']  = "E:/stable-diffusion"
os.environ['SHIVA_CKPT_IMGCODER'] = "E:/MMM"


#####################################################
#           MULTI-VIEW DIFFUSION MODELS             #
#####################################################

CHECKPOINT_ROOT = os.environ.get('SHIVA_CKPT_MVDIFF', None)
if CHECKPOINT_ROOT is not None:
    MULTIVIEW_DIFF_DIR = Path(CHECKPOINT_ROOT)
else:
    MULTIVIEW_DIFF_DIR = Path(__file__).parents[4] / 'checkpoints/multiview'

if os.path.isdir(MULTIVIEW_DIFF_DIR) is False:
    os.makedirs(MULTIVIEW_DIFF_DIR)


MVDIFF_REMOTE_MODELS = {
    "sd15/mvdream": "https://huggingface.co/MVDream/MVDream/resolve/main/sd-v1.5-4view.pt",
    "sd21/mvdream": "https://huggingface.co/MVDream/MVDream/resolve/main/sd-v2.1-base-4view.pt",

    "sd2_original/mvdiffusion": "https://www.dropbox.com/scl/fi/yx9e0lj4fwtm9xh2wlhhg/pano.ckpt?rlkey=kowqygw7vt64r3maijk8klfl0&dl=0",
    "sd2_outpaint/mvdiffusion": "https://www.dropbox.com/scl/fi/3mtj06qx6mxt4eme1oz2r/pano_outpaint.ckpt?rlkey=xat6cwt47lzfjawum05xa5ftq&dl=0",

    "sd15/diffusion360": "https://huggingface.co/archerfmy0831/sd-t2i-360panoimage",
}


MVDIFF_LOCAL_MODELS = {
    "sd15/mvdream": str(MULTIVIEW_DIFF_DIR / "MVDream/sd-v15-4view.pt"),
    "sd21/mvdream": str(MULTIVIEW_DIFF_DIR / "MVDream/sd-v21-base-4view.pt"),

    "sd2_original/mvdiffusion": str(MULTIVIEW_DIFF_DIR / "MVDiffusion/pano.ckpt"),
    "sd2_outpaint/mvdiffusion": str(MULTIVIEW_DIFF_DIR / "MVDiffusion/pano_outpaint.ckpt"),

    "sd15/diffusion360": str(MULTIVIEW_DIFF_DIR / "SD-360panorama"),
}


#############################################
#           (Base) Diffusion Models         #
#############################################

CHECKPOINT_BRANCHES = dict()

for branch in ['sd20', 'sd21']:
    CHECKPOINT_BRANCHES[branch] = os.environ.get(f'SHIVA_CKPT_{branch.upper()}', None)
    if CHECKPOINT_BRANCHES[branch] is not None:
        CHECKPOINT_BRANCHES[branch] = Path(CHECKPOINT_BRANCHES[branch])
    else:
        CHECKPOINT_BRANCHES[branch] = Path(__file__).resolve().parents[4] / f'checkpoints/{branch}'    
    if CHECKPOINT_BRANCHES[branch].exists():
        CHECKPOINT_BRANCHES[branch].mkdir(parents=True, exist_ok=True)

SDIFF_REMOTE_MODELS = {
    "sd21_base"     : "https://huggingface.co/stabilityai/stable-diffusion-2-1-base",
    "sd20_base"     : "https://huggingface.co/stabilityai/stable-diffusion-2-base",
    "sd20_inpaint"  : "https://huggingface.co/stabilityai/stable-diffusion-2-inpainting",
}

SDIFF_LOCAL_MODELS = {
    "sd21_base"     : str(CHECKPOINT_BRANCHES['sd21'] / "checkpoints/sd21_base"),
    "sd20_base"     : str(CHECKPOINT_BRANCHES['sd20'] / "checkpoints/sd20_base"),
    "sd20_inpaint"  : str(CHECKPOINT_BRANCHES['sd20'] / "checkpoints/sd20_inpaint"),
}


#############################################
#           (Auxilliary) Encoders           #
#############################################

IMG_ENCODER_ROOT = os.environ.get('SHIVA_CKPT_IMGCODER', None)
if IMG_ENCODER_ROOT is not None:
    IMG_ENCODER_ROOT = Path(IMG_ENCODER_ROOT)
else:
    IMG_ENCODER_ROOT = Path(__file__).parents[4] / 'checkpoints/multiview'

if os.path.isdir(IMG_ENCODER_ROOT) is False:
    os.makedirs(IMG_ENCODER_ROOT)

REMOTE_IMAGE_ENCODERS = {
    "clip-vit-large-patch14": "https://huggingface.co/openai/clip-vit-large-patch14",
    "clip-vit-h14-laion2B"  : "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
}

LOCAL_IMAGE_ENCODERS = {
    "clip-vit-large-patch14": str(IMG_ENCODER_ROOT / "clip-vit-large-patch14"),
    "clip-vit-h14-laion2B"  : str(IMG_ENCODER_ROOT / "clip-vit-h14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"),
}

