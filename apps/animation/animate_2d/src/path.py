import os
from pathlib import Path


# FIXME: Hardcode to connect to multiple checkpoint directories
os.environ['SHIVA_CKPT_SD15'] = "D:/stable-diffusion/sd-15"
os.environ['SHIVA_CKPT_SD20'] = "D:/stable-diffusion/sd-20"
os.environ['SHIVA_CKPT_SD21'] = "D:/stable-diffusion/sd-21"
os.environ['SHIVA_CKPT_SVD']  = "D:/stable-diffusion/svd"
os.environ['SHIVA_CKPT_ANIMATION'] = "E:/stable-diffusion"


##########################################
#           ANIMATION MODELS             #
##########################################

CHECKPOINT_ROOT = os.environ.get('SHIVA_CKPT_ANIMATION', None)
if CHECKPOINT_ROOT is not None:
    ANIMATION_DIR = Path(CHECKPOINT_ROOT)
else:
    ANIMATION_DIR = Path(__file__).parents[4] / 'checkpoints/animation'

if os.path.isdir(ANIMATION_DIR) is False:
    os.makedirs(ANIMATION_DIR)


ANIMATE_REMOTE_MODELS = {
    # "animateanything"     : "https://huggingface.co/Pupba/animate-anything-512-v1.02",
    "animateanything"     : "https://cloudbook-public-production.oss-cn-shanghai.aliyuncs.com/animation/animate_anything_512_v1.02.tar",
    "animateanything_v2v" : "https://cloudbook-public-production.oss-cn-shanghai.aliyuncs.com/animation/animate_anything_svd_v1.01.tar",
}


ANIMATE_LOCAL_MODELS = {
    "animateanything"     : str(ANIMATION_DIR / "AnimateAnything"),
    "animateanything_v2v" : str(ANIMATION_DIR / "AnimateAnythingSVD"),
}

ANIMATE_LOCAL_MODELS = {m: p.replace('\\', '/') for m, p in ANIMATE_LOCAL_MODELS.items()}


#############################################
#           (Base) Diffusion Models         #
#############################################

CHECKPOINT_BRANCHES = dict()

for branch in ['sd15', 'sd20', 'sd21', 'svd']:
    CHECKPOINT_BRANCHES[branch] = os.environ.get(f'SHIVA_CKPT_{branch.upper()}', None)
    if CHECKPOINT_BRANCHES[branch] is not None:
        CHECKPOINT_BRANCHES[branch] = Path(CHECKPOINT_BRANCHES[branch])
    else:
        CHECKPOINT_BRANCHES[branch] = Path(__file__).resolve().parents[4] / f'checkpoints/{branch}'    
    if CHECKPOINT_BRANCHES[branch].exists():
        CHECKPOINT_BRANCHES[branch].mkdir(parents=True, exist_ok=True)

SDIFF_REMOTE_MODELS = {
    "sd15_base" : "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd21_base" : "https://huggingface.co/stabilityai/stable-diffusion-2-1-base",
    "sd20_base" : "https://huggingface.co/stabilityai/stable-diffusion-2-base",

    # Video Model
    "svd_im2vid"    : "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid",
    "svd_im2vid_xt" : "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt",
}

SDIFF_LOCAL_MODELS = {
    "sd21_base" : str(CHECKPOINT_BRANCHES['sd21'] / "checkpoints/sd21_base"),
    "sd20_base" : str(CHECKPOINT_BRANCHES['sd20'] / "checkpoints/sd20_base"),

    "svd_im2vid"    : str(CHECKPOINT_BRANCHES['svd']  / "svd-img2vid"),     # 14 frames
    "svd_im2vid_xt" : str(CHECKPOINT_BRANCHES['svd']  / "svd-img2vid-xt"),  # 25 frames
}

SDIFF_LOCAL_MODELS = {m: p.replace('\\', '/') for m, p in SDIFF_LOCAL_MODELS.items()}


