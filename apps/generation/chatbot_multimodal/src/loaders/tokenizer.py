from pathlib import Path
from transformers import AutoTokenizer

from ..logging import logger


def load_tokenizer(model_name, tokenizer_dir=None, 
            trust_remote_code: bool = False,
                use_fast_mode: bool = False):
    
    assert model_name in shared.MLM_LOCAL_MODELS.keys()
    path_to_model = Path(shared.MLM_LOCAL_MODELS[model_name])

    if tokenizer_dir:
        path_to_model = Path(tokenizer_dir)

    if path_to_model.exists():
        tokenizer = AutoTokenizer.from_pretrained(
            path_to_model,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_mode,
        )
    else:
        tokenizer = None
    return tokenizer
