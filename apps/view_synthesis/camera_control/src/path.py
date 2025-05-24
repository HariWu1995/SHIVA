import os
from pathlib import Path


# FIXME: Hardcode to connect to multiple checkpoint directories
os.environ['SHIVA_CKPT_SD15'] = "D:/stable-diffusion/sd-15"
os.environ['SHIVA_CKPT_SD20'] = "D:/stable-diffusion/sd-20"
os.environ['SHIVA_CKPT_SD21'] = "D:/stable-diffusion/sd-21"
os.environ['SHIVA_CKPT_SVD'] = "D:/stable-diffusion/svd"
os.environ['SHIVA_CKPT_CAM_CTRL'] = "E:/"
os.environ['SHIVA_CKPT_IMGCODER'] = "E:/MMM"
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
    "animatediff_v3/adaptor" : "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt",
    "animatediff_v3/mmotion" : "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt",
}


ANIMATE_LOCAL_MODELS = {
    "animatediff_v3/adaptor" : str(ANIMATION_DIR / "AnimateDiff/adapter/animatediff-motion-sd15-v3"),
    "animatediff_v3/mmotion" : str(ANIMATION_DIR / "AnimateDiff/lora/animatediff-motion-sd15-v3"),
}

ANIMATE_LOCAL_MODELS = {m: p.replace('\\', '/') for m, p in ANIMATE_LOCAL_MODELS.items()}


###############################################
#           CAMERA-CONTROL MODELS             #
###############################################

CHECKPOINT_ROOT = os.environ.get('SHIVA_CKPT_CAM_CTRL', None)
if CHECKPOINT_ROOT is not None:
    CAMERA_CTRL_DIR = Path(CHECKPOINT_ROOT)
else:
    CAMERA_CTRL_DIR = Path(__file__).parents[4] / 'checkpoints/cam_ctrl'

if os.path.isdir(CAMERA_CTRL_DIR) is False:
    os.makedirs(CAMERA_CTRL_DIR)


CAMCTRL_REMOTE_MODELS = {
        "sd21/seva"     : "https://huggingface.co/stabilityai/stable-virtual-camera",
        "sd15/camctrl"  : "https://huggingface.co/hehao13/CameraCtrl/resolve/main/CameraCtrl.ckpt",
        "svd/camctrl"   : "https://huggingface.co/hehao13/CameraCtrl_SVD_ckpts/resolve/main/CameraCtrl_svd.ckpt",
     "svd_xt/camctrl"   : "https://huggingface.co/hehao13/CameraCtrl_SVD_ckpts/resolve/main/CameraCtrl_svdxt.ckpt",

        # MotionCtrl: 
        # https://huggingface.co/TencentARC/MotionCtrl
         "svd/motionctrl"       : "https://huggingface.co/TencentARC/MotionCtrl/blob/main/motionctrl.pth",
        "lvdm/motionctrl"       : "https://huggingface.co/TencentARC/MotionCtrl/blob/main/motionctrl_svd.ckpt",
    "vidcraft/motionctrl"       : "https://huggingface.co/TencentARC/MotionCtrl/blob/main/motionctrl_videocrafter2_cmcm.ckpt",
    "animdiff/motionctrl_cam"   : "https://huggingface.co/TencentARC/MotionCtrl/resolve/main/motionctrl_animatediff_cmcm.ckpt",
    "animdiff/motionctrl_obj"   : "https://huggingface.co/TencentARC/MotionCtrl/resolve/main/motionctrl_animatediff_omcm.ckpt",
}


CAMCTRL_LOCAL_MODELS = {
           "sd21/seva"          : str(CAMERA_CTRL_DIR / "CameraSeVa"),
           "sd15/camctrl"       : str(CAMERA_CTRL_DIR / "CameraCtrl/CameraCtrl.ckpt"),
            "svd/camctrl"       : str(CAMERA_CTRL_DIR / "CameraCtrl/CameraCtrl_svd.ckpt"),
         "svd_xt/camctrl"       : str(CAMERA_CTRL_DIR / "CameraCtrl/CameraCtrl_svdxt.ckpt"),
         "svd/motionctrl"       : str(CAMERA_CTRL_DIR / "MotionCtrl/motionctrl.pth"),
        "lvdm/motionctrl"       : str(CAMERA_CTRL_DIR / "MotionCtrl/motionctrl_svd.ckpt"),
    "animdiff/motionctrl_cam"   : str(CAMERA_CTRL_DIR / "MotionCtrl/motionctrl_animatediff_cmcm.ckpt"),
    "animdiff/motionctrl_obj"   : str(CAMERA_CTRL_DIR / "MotionCtrl/motionctrl_animatediff_omcm.ckpt"),
    "vidcraft/motionctrl"       : str(CAMERA_CTRL_DIR / "MotionCtrl/motionctrl_videocrafter2_cmcm.ckpt"),
}

CAMCTRL_LOCAL_MODELS = {m: p.replace('\\', '/') for m, p in CAMCTRL_LOCAL_MODELS.items()}


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
    "sd15_base"     : "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd21_base"     : "https://huggingface.co/stabilityai/stable-diffusion-2-1-base",
    "sd20_base"     : "https://huggingface.co/stabilityai/stable-diffusion-2-base",
    "sd20_inpaint"  : "https://huggingface.co/stabilityai/stable-diffusion-2-inpainting",

    # Video Model
    "svd_im2vid"    : "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid",
    "svd_im2vid_xt" : "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt",
}

SDIFF_LOCAL_MODELS = {
    "sd21_base"     : str(CHECKPOINT_BRANCHES['sd21'] / "checkpoints/sd21_base"),
    "sd20_base"     : str(CHECKPOINT_BRANCHES['sd20'] / "checkpoints/sd20_base"),
    "sd20_inpaint"  : str(CHECKPOINT_BRANCHES['sd20'] / "checkpoints/sd20_inpaint"),

    "svd_im2vid"    : str(CHECKPOINT_BRANCHES['svd']  / "svd-img2vid"),     # 14 frames
    "svd_im2vid_xt" : str(CHECKPOINT_BRANCHES['svd']  / "svd-img2vid-xt"),  # 25 frames
}

SDIFF_LOCAL_MODELS = {m: p.replace('\\', '/') for m, p in SDIFF_LOCAL_MODELS.items()}


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
    "clip-vit-large-patch14": "https://huggingface.co/openai/clip-vit-large-patch14",
    "clip-vit-h14-laion2B"  : "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin",
}

LOCAL_IMAGE_ENCODERS = {
    "clip-vit-large-patch14": str(IMG_ENCODER_ROOT / "clip-vit-large-patch14"),
    "clip-vit-h14-laion2B"  : str(IMG_ENCODER_ROOT / "clip-vit-h14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"),
}

LOCAL_IMAGE_ENCODERS = {m: p.replace('\\', '/') for m, p in LOCAL_IMAGE_ENCODERS.items()}

