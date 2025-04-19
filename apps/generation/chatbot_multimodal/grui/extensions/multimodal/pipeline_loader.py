import traceback
from importlib import import_module
from pathlib import Path
from typing import Tuple

import sys

root_dir = str(Path(__file__).resolve().parents[3])
sys.path.append(root_dir)

from grui.extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline

from src import shared
from src.logging import logger


PIPELINE_DIR = Path(__file__).parent / 'pipelines'
PIPELINE_MODULE = "apps.generation.chatbot_multimodal.grui.extensions.multimodal.pipelines.{name}.pipelines"


def _get_available_pipeline_modules():
    modules = [p for p in PIPELINE_DIR.iterdir() if p.is_dir()]
    return [m.name for m in modules if (m / 'pipelines.py').exists()]


def load_pipeline(params: dict) -> Tuple[AbstractMultimodalPipeline, str]:
    pipeline_modules = {}
    pipeline_modules_all = _get_available_pipeline_modules()

    for name in pipeline_modules_all:
        try:
            pipeline_module_name = PIPELINE_MODULE.format(name)
            pipeline_modules[name] = import_module(pipeline_module_name)
        except:
            logger.warning(f'Failed to get multimodal pipelines from {name}')
            logger.warning(traceback.format_exc())

    if shared.args.multimodal_pipeline is not None:
        attr_name = 'get_pipeline'
        for k in pipeline_modules:
            if not hasattr(pipeline_modules[k], attr_name):
                continue
            pipeline = getattr(pipeline_modules[k], attr_name)(shared.args.multimodal_pipeline, params)
            if pipeline is not None:
                return (pipeline, k)
    else:
        model_name = shared.args.model.lower()
        attr_name = 'get_pipeline_from_model_name'
        for k in pipeline_modules:
            if not hasattr(pipeline_modules[k], attr_name):
                continue
            pipeline = getattr(pipeline_modules[k], attr_name)(model_name, params)
            if pipeline is not None:
                return (pipeline, k)

    available = []
    attr_name = 'available_pipelines'
    for k in pipeline_modules:
        if not hasattr(pipeline_modules[k], attr_name):
            continue
        pipelines = getattr(pipeline_modules[k], attr_name)
        available += pipelines

    if shared.args.multimodal_pipeline is not None:
        log = f'Multimodal - ERROR: Failed to load multimodal pipeline "{shared.args.multimodal_pipeline}", available pipelines are: {available}.'
    else:
        log = f'Multimodal - ERROR: Failed to determine multimodal pipeline for model {shared.args.model}, please select one manually using --multimodal-pipeline [PIPELINE]. Available pipelines are: {available}.'
    logger.critical(f'{log} Please specify a correct pipeline, or disable the extension')
    exit()
