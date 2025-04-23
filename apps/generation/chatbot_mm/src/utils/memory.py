import psutil
import gc
import re

import math
import torch
from accelerate.utils import is_npu_available, is_xpu_available


def get_device(use_deepspeed: bool = False):

    if torch.cuda.is_available():
        return torch.device('cuda')

    elif use_deepspeed:
        import deepspeed
        return deepspeed.get_accelerator().current_device_name()

    elif is_xpu_available():
        return torch.device('xpu:0')

    elif is_npu_available():
        return torch.device('npu:0')

    elif torch.backends.mps.is_available():
        return torch.device('mps')

    else:
        return torch.device('cpu')


def get_device_memory(device_id: int = 0, device_type: str = 'cuda'):
    if device_type == 'xpu':
        dev_mem = torch.xpu.get_device_properties(device_id).total_memory
    elif device_type == 'npu':
        dev_mem = torch.npu.get_device_properties(device_id).total_memory
    else:
        dev_mem = torch.cuda.get_device_properties(device_id).total_memory
    return math.floor(dev_mem / (1024 * 1024)) 


def get_all_device_memory():
    total_mem = dict()
    if is_xpu_available():
        for i in range(torch.xpu.device_count()):
            total_mem[f'xpu:{i}'] = get_device_memory(i, 'xpu')
    elif is_npu_available():
        for i in range(torch.npu.device_count()):
            total_mem[f'npu:{i}'] = get_device_memory(i, 'npu')
    else:
        for i in range(torch.cuda.device_count()):
            total_mem[f'cuda:{i}'] = get_device_memory(i, 'cuda')
    return total_mem


def get_max_memory_info():
    max_memory = get_all_device_memory()
    max_memory['cpu'] = get_default_cpu_mem()
    return max_memory


def get_default_cpu_mem():
    total_cpu_mem = math.floor(psutil.virtual_memory().total / (1024 * 1024))
    return total_cpu_mem


def clear_torch_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_xpu_available():
        torch.xpu.empty_cache()
    elif is_npu_available():
        torch.npu.empty_cache()
    elif torch.backends.mps.is_available():
        if hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()


if __name__ == "__main__":
    import json
    print(json.dumps(get_max_memory_info(), indent=4))

