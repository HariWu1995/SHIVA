from pathlib import Path

from ..logging import logger, pprinter
from ..utils import get_max_memory_info
from .. import shared
from .utils import config_loader_tensorrt as config_loader


def TensorRT_LLM_loader(model_name):
    
    assert model_name in shared.MLM_LOCAL_MODELS.keys()
    path_to_model = Path(shared.MLM_LOCAL_MODELS[model_name])

    shared.model_loader_config = config_loader

    from ..models.tensorrt_llm import TensorRTLLModel
    model = TensorRTLLModel.from_pretrained(path_to_model, loader_config)
    return model

