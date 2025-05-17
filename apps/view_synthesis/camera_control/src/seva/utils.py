import os
from typing import Any, Dict, List, Optional, Tuple

import random as rd
import numpy as np

import torch
import safetensors.torch as safetorch
from huggingface_hub import hf_hub_download

from .. import shared


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, _, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def seed_everything(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def randomize_seed():
    return rd.randint(1, np.iinfo(int).max)


def onload(model):
    model.to(device=shared.device)


def offload(model):
    if shared.low_vram:
        model.cpu()
        torch.cuda.empty_cache()

