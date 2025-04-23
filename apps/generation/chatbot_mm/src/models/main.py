import os
from glob import glob
from time import time

from .. import shared
from ..path import MLM_LOCAL_MODELS, MULTIMODAL_ALLOWED
from ..utils import clear_torch_cache
from ..logging import logger


MODELLIB_LOADERS = ['transformers', 'llamacpp']


def find_model_type(model_name: str):
    if len(glob(
        MLM_LOCAL_MODELS[model_name]+'/*.gguf'
    )) > 0:
        return 'llamacpp'
    elif any([
        os.path.isfile(os.path.join(MLM_LOCAL_MODELS[model_name], x)) 
        for x in ["model.safetensors", "pytorch_model.bin"]
    ]):
        return 'transformers'
    return 'unknown'


def load_models(model_name: str, model_type: str = 'auto'):

    if model_type == 'auto':
        model_type = find_model_type(model_name)

    assert model_name in MLM_LOCAL_MODELS
    assert model_type in MODELLIB_LOADERS

    shared.model_name = model_name
    shared.model_type = model_type

    # LlamaCpp
    if model_type == 'llamacpp':
        from .llamacpp import load_model, generate_response, build_prompt, generation_config, parse_response
    
    # Transformers
    elif 'deepseek' in model_name:
        if 'VL' in model_name:
            raise ImportError("Deepseek Vision-Language is not implemented yet!")
        from .transformers.deepseek import load_model, generate_response, build_prompt, generation_config, parse_response

    elif 'qwen' in model_name:
        if 'VL' in model_name:
            from .transformers.qwen_vl import load_model, generate_response, build_prompt, generation_config, parse_response
        else:
            from .transformers.qwen import load_model, generate_response, build_prompt, generation_config, parse_response

    elif 'moondream' in model_name:
        from .transformers.moondream import load_model, generate_response, build_prompt, generation_config, parse_response

    else:
        from .transformers.common import load_model, generate_response, build_prompt, generation_config, parse_response

    # Load generator and tokenizer / processor
    if model_type == "transformers":
        model_config = dict()
    else:
        model_config = dict(
                n_ctx = shared.model_args.max_seq_len, 
            n_threads = shared.model_args.n_threads, 
         n_gpu_layers = shared.model_args.n_gpu_layers, 
              verbose = shared.model_args.verbose
        )

    tokenizer, generator = load_model(model_name, **model_config)

    # Store generation variables
    shared.postprocess_fn    = parse_response
    shared.preparation_fn    = build_prompt
    shared.generation_fn     = generate_response
    shared.generation_config = generation_config
    
    # Multi-modal allowance
    shared.multimodals = MULTIMODAL_ALLOWED[model_name]
    
    return tokenizer, generator


def reload_model():
    return load_models(shared.model_name, shared.model_type)


def unload_model(keep_model_info=False):

    # Keep for reload
    if not keep_model_info:
        shared.model_name = None
        shared.model_type = None

    shared.generator = None
    shared.tokenizer = None
    shared.multimodals = None

    shared.postprocess_fn    = None
    shared.preparation_fn    = None
    shared.generation_fn     = None
    shared.generation_config = None

    clear_torch_cache()


def unload_model_if_idle():

    idle_timeout = shared.model_args.idle_timeout * 60
    logger.info(f"Setting timeout ({idle_timeout} seconds) to unload model if inactivity.")

    while True:
        shared.generation_lock.acquire()
        try:
            if (time() - shared.generation_last_time) > idle_timeout:
                if shared.generator is not None:
                    logger.info("Unloading the model for inactivity.")
                    unload_model(keep_model_name=True)
        finally:
            shared.generation_lock.release()
        time.sleep(idle_timeout//2)


