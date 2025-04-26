import os
import time
import torch

from .utils import get_device
from .path import (
    VAENCODERS,
    LORA_LONGLIST,
    MODEL_EXTENSIONS,
    IMAGENE_REMOTE_MODELS, IMAGENE_REMOTE_CTRLNETS, IMAGENE_REMOTE_ADAPTERS, 
    IMAGENE_LOCAL_MODELS,  IMAGENE_LOCAL_CTRLNETS,  IMAGENE_LOCAL_ADAPTERS,  
)


# Device variables
device = get_device()
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32
low_vram = os.environ.get("LOW_VRAM", True)

# Model variables
model_extensions = MODEL_EXTENSIONS
model = None
model_name = None
lora_names = []

# Generation variables
generation_last_time = time.time()
generation_lock = None
generation_config = {}

# Misc.
stop_everything = False
multi_user = False
verbose = True

