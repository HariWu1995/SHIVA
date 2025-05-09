import os
from pathlib import Path


CHECKPOINT_ROOT = os.environ.get('SHIVA_CKPT_ROOT', None)
if CHECKPOINT_ROOT is not None:
    SUPERRES_DIR = Path(CHECKPOINT_ROOT) / 'superres'
else:
    SUPERRES_DIR = Path(__file__).parents[4] / 'checkpoints/superres'

if os.path.isdir(SUPERRES_DIR) is False:
    os.makedirs(SUPERRES_DIR)


SUPERRES_LOCAL_MODELS = {
    "realesgan_x2+" : str(SUPERRES_DIR / "RealESRGAN_x2plus.pth"),
    "realesgan_x4+" : str(SUPERRES_DIR / "RealESRGAN_x4plus.pth"),

    # Face
    "restoreformer" : str(SUPERRES_DIR / "RestoreFormer.pth"),
    "gfpgan_v13"    : str(SUPERRES_DIR / "GFPGANv1.3.pth"),
    "gfpgan_v14"    : str(SUPERRES_DIR / "GFPGANv1.4.pth"),
}


SUPERRES_REMOTE_MODELS = {
    "realesgan_x4+" : 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    "realesgan_x2+" : 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    
    "gfpgan_v13"    : 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
    "gfpgan_v14"    : 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    "restoreformer" : 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth',
}



