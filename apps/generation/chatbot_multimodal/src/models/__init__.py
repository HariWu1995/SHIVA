import gc
import time


last_generation_time = time.time()


def clear_model_cache():
    gc.collect()
    if shared.args.cpu:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_xpu_available():
        torch.xpu.empty_cache()
    elif is_npu_available():
        torch.npu.empty_cache()
    elif torch.backends.mps.is_available():
        if hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()

