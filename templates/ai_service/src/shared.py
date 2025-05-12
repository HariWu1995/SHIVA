import os
import time
import torch


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Device variables
device = get_device()
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32
low_vram = os.environ.get("LOW_VRAM", True)
offloaded = False

# Model variables
model = None
model_name = None

# Generation variables
generation_last_time = time.time()
generation_config = {}
generation_lock = None

# Misc.
stop_everything = False
multi_user = False
verbose = True

