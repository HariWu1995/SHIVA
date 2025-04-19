import os
import yaml
from copy import deepcopy

import torch
from transformers import is_torch_xpu_available, is_torch_npu_available

from pathlib import Path, PurePath
from collections import OrderedDict

from .logging import logger
from .path import MLM_LOCAL_MODELS, MULTIMODAL_LM_DIR
from .utils import load_config


def get_device(use_deepspeed: bool = False):

    if torch.cuda.is_available():
        return torch.device('cuda')

    elif use_deepspeed:
        import deepspeed
        return deepspeed.get_accelerator().current_device_name()

    elif torch.backends.mps.is_available():
        return torch.device('mps')

    elif is_torch_xpu_available():
        return torch.device('xpu:0')

    elif is_torch_npu_available():
        return torch.device('npu:0')

    else:
        return torch.device('cpu')


#####################################
#              Variables            #
#####################################

# Model variables
loader = None
model = None
model_name = 'None'
tokenizer = None
is_seq2seq = False
lora_names = []

# Generation variables
stop_everything = False
generation_lock = None
processing_message = '*Is typing...*'


#############################################
#                 Arguments                 #
#   https://github.com/vladmandic/automatic #
#############################################

arg_config_path = Path(__file__).resolve().parent / "config/arguments.yaml"
args = load_config(arg_config_path, return_type="namespace")
args.model_dir = str(MULTIMODAL_LM_DIR)
args.lora_dir = str(MULTIMODAL_LM_DIR)

args_default = deepcopy(args)
args_provided = []
args_deprecated = [
    'cache_4bit',
    'cache_8bit',
    'triton',
    'inject_fused_mlp',
    'use_cuda_fp16',
    'disable_exllama',
    'disable_exllamav2',
    'wbits',
    'groupsize',
]

#########################################
#       Model-specific settings         #
#########################################

model_config_path = Path(__file__).resolve().parent / "config/model_default.yaml"
model_config = load_config(model_config_path, return_type="ordereddict")

user_config_path = Path(__file__).resolve().parent / "config/model_user.yaml"
user_config = load_config(user_config_path, return_type="ordereddict")

model_loader_config = None
