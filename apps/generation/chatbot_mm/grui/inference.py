"""
Message Format:
    +   text: {'role': 'user', 'content':  text  , 'metadata': None, 'options': None}
    + others: {'role': 'user', 'content': (path,), 'metadata': None, 'options': None}
"""
import time
from copy import deepcopy

from ..src import shared
from ..src.logging import logger


def run_inference(state: dict, history: list):

    # Overwrite generation config
    gen_config = shared.generation_config
    for k, v in state.items():
        if k in gen_config:
            gen_config[k] = v

    assert shared.generator is not None, "Model is not loaded!"

    # Generation
    inputs = deepcopy(history)
    if callable(shared.preparation_fn):
        inputs = shared.preparation_fn(inputs)
    if callable(shared.generation_fn):
        output = shared.generation_fn(shared.generator, shared.tokenizer, **inputs, **gen_config)
    if callable(shared.postprocess_fn):
        output = shared.postprocess_fn(output, role='assistant')

    # Aggregation
    if isinstance(output, dict):
        history.append(output)
    elif isinstance(output, list):
        history.extend(output)      
    return history


