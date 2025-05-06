import os
import time
import torch

from .utils import get_device, MODEL_EXTENSIONS
from .models.path import MVDIFF_LOCAL_MODELS, MVDIFF_REMOTE_MODELS


# Device variables
device = get_device()
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32
low_vram = os.environ.get("LOW_VRAM", True)
offloaded = False

# Model variables
model = None
model_name = None
model_extensions = MODEL_EXTENSIONS

# Generation variables
generation_last_time = time.time()
generation_lock = None
generation_config = {}

# Misc.
stop_everything = False
multi_user = False
verbose = True

