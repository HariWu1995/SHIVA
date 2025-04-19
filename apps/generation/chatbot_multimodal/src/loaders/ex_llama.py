from pathlib import Path

from ..logging import logger, pprinter
from ..utils import get_max_memory_info
from .. import shared
from .utils import config_loader_exllama as config_loader


def ExLlamav3_HF_loader(model_name):
    
    assert model_name in shared.MLM_LOCAL_MODELS.keys()
    path_to_model = Path(shared.MLM_LOCAL_MODELS[model_name])

    shared.model_loader_config = config_loader
    
    from ..models.exllamav3_hf import Exllamav3HF
    return Exllamav3HF.from_pretrained(path_to_model, config_loader)


def ExLlamav2_HF_loader(model_name):
    
    assert model_name in shared.MLM_LOCAL_MODELS.keys()
    path_to_model = Path(shared.MLM_LOCAL_MODELS[model_name])

    shared.model_loader_config = config_loader

    from ..models.exllamav2_hf import Exllamav2HF
    return Exllamav2HF.from_pretrained(path_to_model, config_loader)


def ExLlamav2_loader(model_name):
    
    assert model_name in shared.MLM_LOCAL_MODELS.keys()
    path_to_model = Path(shared.MLM_LOCAL_MODELS[model_name])

    shared.model_loader_config = config_loader

    from ..models.exllamav2 import Exllamav2Model
    return Exllamav2Model.from_pretrained(path_to_model, config_loader)

