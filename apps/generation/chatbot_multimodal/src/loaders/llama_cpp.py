from pathlib import Path

from ..logging import logger, pprinter
from ..utils import get_max_memory_info
from .. import shared
from .utils import config_loader_llamacpp as config_loader


def llamacpp_loader(model_name):
    from ..models.llamacpp_model import LlamaCppModel
    
    assert model_name in shared.MLM_LOCAL_MODELS.keys()
    path_to_model = Path(shared.MLM_LOCAL_MODELS[model_name])

    if path_to_model.is_file():
        model_file = path_to_model
    else:
        model_file = sorted(path_to_model.glob('*.gguf'))[0]

    logger.info(f"llama.cpp weights detected: \"{model_file}\"")
    model, tokenizer = LlamaCppModel.from_pretrained(model_file)

    shared.model_loader_config = config_loader
    return model, tokenizer


def llamacpp_HF_loader(model_name):
    from ..models.llamacpp_hf import LlamacppHF
    
    assert model_name in shared.MLM_LOCAL_MODELS.keys()
    path_to_model = Path(shared.MLM_LOCAL_MODELS[model_name])

    tokenizer_dir = config_loader.tokenizer_dir
    if tokenizer_dir:
        logger.info(f'Using tokenizer from: \"{tokenizer_dir}\"')
    else:
        # Check if a HF tokenizer is available for the model
        if all((path_to_model / file).exists() 
                            for file in ['tokenizer_config.json']):
            logger.info(f'Using tokenizer from: \"{str(path_to_model)}\"')
        else:
            logger.error("Could not load the model because missing `tokenizer_config.json`.")
            return None, None

    model = LlamacppHF.from_pretrained(model_name)
    shared.model_loader_config = config_loader

    if tokenizer_dir:
        from .tokenizer import load_tokenizer
        tokenizer = load_tokenizer(model_name, tokenizer_dir, config_loader.trust_remote_code, 
                                                              config_loader.use_fast_tokenizer)
        return model, tokenizer
    else:
        return model

