from ast import literal_eval

import copy
import html
import time
import traceback

import numpy as np
import torch

from .. import shared, logger, pprinter
from ..cache import process_llamacpp_cache
from ..callbacks import Stream

from .. import models
from ..loaders import load_model

from .. import sampler
sampler.hijack_samplers()

from .utils import set_manual_seed, print_prompt, formatted_outputs
from .default import generate_reply_HF, encode, decode
from .customized import generate_reply_custom


def generate_reply(*args, **kwargs):
    if shared.args.idle_timeout > 0 \
    and shared.model_name not in [None, 'None']\
    and shared.model is None:
        shared.model, \
        shared.tokenizer = load_model(shared.model_name)

    shared.generation_lock.acquire()
    try:
        for result in _generate_reply(*args, **kwargs):
            yield result
    finally:
        models.last_generation_time = time.time()
        shared.generation_lock.release()


def _generate_reply(question, state, stopping_strings=None, is_chat=False, escape_html=False, for_ui=False):

    if shared.model.__class__.__name__ in ['LlamaCppModel', 'Exllamav2Model', 'TensorRTLLMModel']:
        generate_func = generate_reply_custom
    else:
        generate_func = generate_reply_HF
    if generate_func != generate_reply_HF and shared.args.verbose:
        logger.info("PROMPT=")
        print_prompt(question)

    # Prepare the input
    original_question = question

    # Find the stopping strings
    all_stop_strings = []
    for st in (stopping_strings, state['custom_stopping_strings']):
        if type(st) is str:
            st = literal_eval(f"[{st}]")

        if type(st) is list and len(st) > 0:
            all_stop_strings += st

    shared.stop_everything = False
    seed = set_manual_seed(state['seed'])
    last_update = -1
    reply = ''
    is_stream = state['stream']
    if len(all_stop_strings) > 0 and not state['stream']:
        state = copy.deepcopy(state)
        state['stream'] = True

    min_update_interval = 0
    if state.get('max_updates_second', 0) > 0:
        min_update_interval = 1 / state['max_updates_second']

    # Generate
    for reply in generate_func(question, original_question, seed, state, stopping_strings, is_chat=is_chat):
        reply, stop_found = apply_stopping_strings(reply, all_stop_strings)
        if escape_html:
            reply = html.escape(reply)

        if is_stream:
            cur_time = time.time()

            # Limit number of tokens/second to make text readable in real time
            if state['max_tokens_second'] > 0:
                diff = 1 / state['max_tokens_second'] - (cur_time - last_update)
                if diff > 0:
                    time.sleep(diff)

                last_update = time.time()
                yield reply

            # Gradio updates to avoid lag
            # FastAPI updates are not limited
            else:
                if cur_time - last_update > min_update_interval:
                    last_update = cur_time
                    yield reply

                yield reply

        if stop_found or (state['max_tokens_second'] > 0 and shared.stop_everything):
            break

    yield reply


def get_encoded_length(prompt, length_after_extensions=None):
    if length_after_extensions is not None:
        return length_after_extensions
    return len(encode(prompt)[0])


def get_token_ids(prompt):
    tokens = encode(prompt)[0]
    decoded_tokens = [shared.tokenizer.decode([i]) for i in tokens]

    output = ''
    for row in list(zip(tokens, decoded_tokens)):
        output += f"{str(int(row[0])).ljust(5)}  -  {repr(row[1])}\n"
    return output


def generate_reply_wrapper(question, state, stopping_strings=None):
    """
    Returns formatted outputs for the UI
    """
    reply = question if not shared.is_seq2seq else ''
    yield formatted_outputs(reply, shared.model_name)

    for reply in generate_reply(question, state, stopping_strings, is_chat=False, escape_html=True, for_ui=True):
        if not shared.is_seq2seq:
            reply = question + reply

        yield formatted_outputs(reply, shared.model_name)


def stop_everything_event():
    shared.stop_everything = True


def apply_stopping_strings(reply, all_stop_strings):
    stop_found = False
    for string in all_stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou:"
        # is completed, trim it
        for string in all_stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
                else:
                    continue
            break

    return reply, stop_found

