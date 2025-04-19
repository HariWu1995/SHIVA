from pathlib import Path

from ..logging import logger, pprinter
from ..utils import get_max_memory_info
from .. import shared
from .utils import config_loader_hflibs as config_loader

from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import is_ccl_available, is_npu_available, is_xpu_available

import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig


def huggingface_loader(model_name, low_cpu_mem_usage: bool = True):
    
    assert model_name in shared.MLM_LOCAL_MODELS.keys()
    path_to_model = Path(shared.MLM_LOCAL_MODELS[model_name])

    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path = path_to_model, 
        trust_remote_code = config_loader.trust_remote_code)

    if 'chatglm' in model_name.lower():
        from transformers import AutoModel as LoaderClass

    elif config.to_dict().get('is_encoder_decoder', False):
        from transformers import AutoModelForSeq2SeqLM as LoaderClass
        shared.is_seq2seq = True

    else:
        from transformers import AutoModelForCausalLM as LoaderClass

    if torch.cuda.is_bf16_supported() and config_loader.use_bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    params = dict(
        torch_dtype = torch_dtype,
        low_cpu_mem_usage = low_cpu_mem_usage,
        use_safetensors = config_loader.use_safetensors,
        trust_remote_code = config_loader.trust_remote_code,
        attn_implementation = config_loader.attn_implementation,,
        use_flash_attention_2 = config_loader.use_flash_attention_2,
    )

    # Determine if we should use default loading
    use_default_loading = not any([
        config_loader.cpu,
        config_loader.load_in_8bit,
        config_loader.load_in_4bit,
        config_loader.auto_devices,
        config_loader.disk,
        config_loader.use_deepspeed,
        config_loader.gpu_memory is not None,
        config_loader.cpu_memory is not None,
        config_loader.compress_pos_emb > 1,
        config_loader.alpha_value > 1,
    ])

    # Load the model without any special settings
    if use_default_loading:
        logger.info("TRANSFORMERS_PARAMS=")
        pprinter.pprint(params)

        model = LoaderClass.from_pretrained(path_to_model, **params)
        if not (hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit):
            device = shared.get_device(config_loader.use_deepspeed)
            model = model.to(device)

    # DeepSpeed ZeRO-3
    elif config_loader.use_deepspeed:
        import deepspeed
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, HfDeepSpeedConfig
        
        if is_deepspeed_zero3_enabled():
            logger.info('DeepSpeed ZeRO-3 is enabled.')

        model = LoaderClass.from_pretrained(
            path_to_model,
            torch_dtype=params['torch_dtype'],
            trust_remote_code=params.get('trust_remote_code')
        )

        from .deepspeed import update_deepspeed_config
        deepspeed_config = update_deepspeed_config(config_loader)
        hfdeepspeed_config = HfDeepSpeedConfig(deepspeed_config)

        model = deepspeed.initialize(
            model=model,
            config_params=deepspeed_config,
            model_parameters=None,
            optimizer=None,
            lr_scheduler=None
        )[0]

        model.module.eval()  # Inference

    # Load with quantization and/or offloading
    else:
        if not any([config_loader.cpu, 
                    torch.cuda.is_available(),
            torch.backends.mps.is_available(), is_xpu_available()]):
            logger.warning('No GPU has been detected. Falling back to CPU mode.')
            config_loader.cpu = True
            shared.args.cpu = True

        if config_loader.cpu:
            params['torch_dtype'] = torch.float32
        else:
            params['device_map'] = 'auto'
            if x := get_max_memory_info(config_loader):
                params['max_memory'] = x

            if config_loader.load_in_4bit:
                # See https://github.com/huggingface/transformers/pull/23479/files
                # and https://huggingface.co/blog/4bit-transformers-bitsandbytes
                quantization_config_params = {
                    'load_in_4bit': True,
                    'bnb_4bit_compute_dtype': eval(f"torch.{config_loader.compute_dtype}") if config_loader.compute_dtype in ["bfloat16", "float16", "float32"] else None,
                    'bnb_4bit_quant_type': config_loader.quant_type,
                    'bnb_4bit_use_double_quant': config_loader.use_double_quant,
                    'llm_int8_enable_fp32_cpu_offload': True
                }
                params['quantization_config'] = BitsAndBytesConfig(**quantization_config_params)

            elif config_loader.load_in_8bit:
                if config_loader.auto_devices or config_loader.gpu_memory:
                    params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
                else:
                    params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)

                if len(params.get('max_memory')) > 0:
                    with init_empty_weights():
                        model = LoaderClass.from_config(config, trust_remote_code=params.get('trust_remote_code'))

                    model.tie_weights()
                    params['device_map'] = infer_auto_device_map(
                        model,
                        dtype=torch.int8,
                        max_memory=params.get('max_memory'),
                        no_split_module_classes=model._no_split_modules
                    )

            if config_loader.disk:
                params['offload_folder'] = config_loader.disk_cache_dir

        if config_loader.compress_pos_emb > 1:
            params['rope_scaling'] = {'type': 'linear', 'factor': config_loader.compress_pos_emb}
        elif config_loader.alpha_value > 1:
            params['rope_scaling'] = {'type': 'dynamic', 'factor': config_loader.alpha_value}

        logger.info("TRANSFORMERS_PARAMS=")
        pprinter.pprint(params)
        model = LoaderClass.from_pretrained(path_to_model, **params)

    if config_loader.torch_compile:
        model = torch.compile(model)

    shared.model_loader_config = config_loader
    return model


