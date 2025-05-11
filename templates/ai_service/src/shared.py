import os
import time
import torch


# Device variables
device = get_device()
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32
low_vram = os.environ.get("LOW_VRAM", True)
offloaded = False

# Model variables
model = None
model_name = None
model_extensions = []

# Generation variables
generation_last_time = time.time()
generation_config = {}
generation_lock = None

# Misc.
stop_everything = False
multi_user = False
verbose = True

