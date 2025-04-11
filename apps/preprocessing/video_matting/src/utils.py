import os
from pathlib import Path


CHECKPOINT_ROOT = os.environ.get('SHIVA_CKPT_ROOT', None)
if CHECKPOINT_ROOT is not None:
    RVM_DIR = Path(CHECKPOINT_ROOT) / 'rvm'
else:
    RVM_DIR = Path(__file__).parents[4] / 'checkpoints/rvm'

rvm_models = {
    'mobilenetv3': str(RVM_DIR / 'rvm_mobilenetv3.pth'),
       'resnet50': str(RVM_DIR / 'rvm_resnet50.pth'),
}


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)
