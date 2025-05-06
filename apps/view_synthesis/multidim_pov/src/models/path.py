import os
from pathlib import Path


# FIXME: Hardcode to connect to multiple checkpoint directories
os.environ['SHIVA_CKPT_MVDIFF']  = "E:/stable-diffusion"
os.environ['SHIVA_CKPT_IMGCODER'] = "E:/MMM"


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
}


MVDIFF_LOCAL_MODELS = {
    "sd15/mvdream": str(MULTIVIEW_DIFF_DIR / "MVDream/sd-v15-4view.pt"),
    "sd21/mvdream": str(MULTIVIEW_DIFF_DIR / "MVDream/sd-v21-base-4view.pt"),

    "sd2_original/mvdiffusion": str(MULTIVIEW_DIFF_DIR / "MVDiffusion/pano.ckpt"),
    "sd2_outpaint/mvdiffusion": str(MULTIVIEW_DIFF_DIR / "MVDiffusion/pano_outpaint.ckpt"),
}


# Check if text-/image-encoder exists
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

