import os
import torch
from accelerate.utils import is_ccl_available, is_npu_available, is_xpu_available


def update_deepspeed_config(args, local_rank = None, world_size = None):
    import deepspeed
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

    # Distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0")) if not local_rank else local_rank
    world_size = int(os.getenv("WORLD_SIZE", "1")) if not world_size else world_size

    if is_xpu_available() and is_ccl_available():
        torch.xpu.set_device(local_rank)
        deepspeed.init_distributed(backend="ccl")

    elif is_npu_available():
        torch.npu.set_device(local_rank)
        deepspeed.init_distributed(dist_backend="hccl")

    else:
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed()

    deepspeed_config = generate_deepspeed_config(args.bf16, 1 * world_size, args.nvme_offload_dir)
    return deepspeed_config


def generate_deepspeed_config(ds_bf16, train_batch_size, nvme_offload_dir):
    '''
    DeepSpeed configuration
        https://huggingface.co/docs/transformers/main_classes/deepspeed
    '''
    if nvme_offload_dir:
        ds_config = {
            "fp16": {
                "enabled": not ds_bf16,
            },
            "bf16": {
                "enabled": ds_bf16,
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "nvme",
                    "nvme_path": nvme_offload_dir,
                    "pin_memory": True,
                    "buffer_count": 5,
                    "buffer_size": 1e9,
                    "max_in_cpu": 1e9
                },
                "overlap_comm": True,
                "reduce_bucket_size": "auto",
                "contiguous_gradients": True,
                "sub_group_size": 1e8,
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": "auto",
                "stage3_max_reuse_distance": "auto",
            },
            "aio": {
                "block_size": 262144,
                "queue_depth": 32,
                "thread_count": 1,
                "single_submit": False,
                "overlap_events": True
            },
            "steps_per_print": 2000,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
        }
    else:
        ds_config = {
            "fp16": {
                "enabled": not ds_bf16,
            },
            "bf16": {
                "enabled": ds_bf16,
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": "auto",
                "stage3_max_reuse_distance": "auto",
            },
            "steps_per_print": 2000,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
        }

    return ds_config

