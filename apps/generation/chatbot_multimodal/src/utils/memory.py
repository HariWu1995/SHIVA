import math
import torch
from accelerate.utils import is_ccl_available, is_npu_available, is_xpu_available


def get_max_memory_info(args):
    max_memory = {}
    max_cpu_memory = args.cpu_memory.strip() if args.cpu_memory is not None else '64GiB'

    if args.gpu_memory:
        memory_map = list(map(lambda x: x.strip(), args.gpu_memory))
        for i in range(len(memory_map)):
            max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
        max_memory['cpu'] = f'{max_cpu_memory}GiB' if not re.match('.*ib$', max_cpu_memory.lower()) else max_cpu_memory

    # If --auto-devices is provided standalone,
    # try to get a reasonable value for the maximum memory of device :0
    elif args.auto_devices:
        if is_xpu_available():
            total_mem = (torch.xpu.get_device_properties(0).total_memory / (1024 * 1024))
        else:
            total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))

        suggestion = round((total_mem - 1000) / 1000) * 1000
        if total_mem - suggestion < 800:
            suggestion -= 1000

        suggestion = int(round(suggestion / 1000))
        logger.warning(f"Auto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors. You can manually set other values.")

        max_memory[0] = f'{suggestion}GiB'
        max_memory['cpu'] = f'{max_cpu_memory}GiB' if not re.match('.*ib$', max_cpu_memory.lower()) else max_cpu_memory

    return max_memory


def get_device_memory(device_id: int = 0, device_type: str = 'cuda'):
    if device_type == 'xpu':
        dev_mem = torch.xpu.get_device_properties(device_id).total_memory
    elif device_type == 'npu':
        dev_mem = torch.npu.get_device_properties(device_id).total_memory
    else:
        dev_mem = torch.cuda.get_device_properties(device_id).total_memory
    return math.floor(dev_mem / (1024 * 1024)) 


def get_all_device_memory():

    total_mem = []
    if is_xpu_available():
        for i in range(torch.xpu.device_count()):
            total_mem.append(get_device_memory(i, 'xpu'))
    elif is_npu_available():
        for i in range(torch.npu.device_count()):
            total_mem.append(get_device_memory(i, 'npu'))
    else:
        for i in range(torch.cuda.device_count()):
            total_mem.append(get_device_memory(i, 'cuda'))
    return total_mem


