import os
import time
import torch

from .utils import get_device
from .models.path import MVDIFF_LOCAL_MODELS, MVDIFF_REMOTE_MODELS


# Device variables
device = get_device()
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32
low_vram = os.environ.get("LOW_VRAM", True)
offloaded = False

# Model variables
pipeline = None
model_base = None
model_name = None
model_refine = None

# Generation variables
generation_last_time = time.time()
generation_config = {}
generation_lock = None

# Misc.
stop_everything = False
multi_user = False
verbose = True

# Preset
positive_prompt = 'photorealistic, artstation, best quality, high resolution, 8k rendering'

negative_prompt = 'worst quality, low quality, logo, text, watermark, monochrome, blur'\
                ', deformed, disfigured, distorted, human, person, small object, '\
                ', complex texture, complex lighting, oversaturated'

