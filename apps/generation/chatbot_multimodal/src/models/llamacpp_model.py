import re
from functools import partial

import numpy as np
import torch

from .. import shared
from ..logging import logger
from ..callbacks import Iteratorize
from .llama_cpp_python_hijack import llama_cpp_lib


llamacpp_quant_mapping = {
    'f32': 0,
    'fp16': 1,
    'q4_0': 2,
    'q4_1': 3,
    'q5_0': 6,
    'q5_1': 7,
    'q8_0': 8,
    'q8_1': 9,
    'q2_k': 10,
    'q3_k': 11,
    'q4_k': 12,
    'q5_k': 13,
    'q6_k': 14,
    'q8_k': 15,
    'iq4_nl': 20,
    'bf16': 30,
}

llamacpp_valid_cache_types = {'fp16', 'q8_0', 'q4_0'}


class LlamaCppModel:

    def __init__(self, loader_config):
        self.initialized = False
        self.loader_config = loader_config
        self.grammar_string = ''
        self.grammar = None

    def __del__(self):
        del self.model

    @classmethod
    def from_pretrained(self, path):

        Llama      = llama_cpp_lib(self.loader_config).Llama
        LlamaCache = llama_cpp_lib(self.loader_config).LlamaCache

        cache_capacity = 0
        if self.loader_config.cache_capacity is not None:
            if 'GiB' in self.loader_config.cache_capacity:
                cache_capacity = int(re.sub('[a-zA-Z]', '', self.loader_config.cache_capacity)) * 1000 * 1000 * 1000
            elif 'MiB' in self.loader_config.cache_capacity:
                cache_capacity = int(re.sub('[a-zA-Z]', '', self.loader_config.cache_capacity)) * 1000 * 1000
            else:
                cache_capacity = int(self.loader_config.cache_capacity)
        if cache_capacity > 0:
            logger.info("Cache capacity is " + str(cache_capacity) + " bytes")

        if self.loader_config.tensor_split is None \
        or self.loader_config.tensor_split.strip() == '':
            tensor_split_list = None
        else:
            tensor_split_list = [float(x) for x in self.loader_config.tensor_split.strip().split(",")]

        params = {
            'model_path': str(path),
            'n_ctx': self.loader_config.n_ctx,
            'n_threads': self.loader_config.n_threads or None,
            'n_threads_batch': self.loader_config.n_threads_batch or None,
            'n_batch': self.loader_config.n_batch,
            'use_mmap': self.loader_config.use_mmap,
            'use_mlock': self.loader_config.use_mlock,
            'mul_mat_q': self.loader_config.mul_mat_q,
            'numa': self.loader_config.numa,
            'n_gpu_layers': self.loader_config.n_gpu_layers,
            'rope_freq_base': self.loader_config.rope_freq_base,
            'rope_freq_scale': 1.0 / self.loader_config.compress_pos_emb,
            'tensor_split': tensor_split_list,
            'offload_kqv': self.loader_config.offload_kqv,
            'split_mode': self.loader_config.split_mode,
            'flash_attn': self.loader_config.flash_attn,
        }

        if self.loader_config.cache_type != 'fp16':
            params["type_k"] = get_llamacpp_cache_type_for_string(shared.args.cache_type)
            params["type_v"] = get_llamacpp_cache_type_for_string(shared.args.cache_type)

        result = self()
        try:
            result.model = Llama(**params)
        except Exception as e:
            error_message = (
                f"Failed loading the model. **This usually happens due to lack of memory**. Try these steps:\n"
                f"1. Reduce the context length `n_ctx` (currently {self.loader_config.n_ctx})."
                f"{' Try a lower value like 4096.' if self.loader_config.n_ctx > 4096 else '.'}\n"
                f"2. Lower the `n-gpu-layers` value (currently {self.loader_config.n_gpu_layers})."
            )
            raise type(e)(error_message) from e

        if cache_capacity > 0:
            result.model.set_cache(LlamaCache(capacity_bytes=cache_capacity))

        # This is ugly, but the model and the tokenizer are the same object in this library.
        return result, result

    def encode(self, string):
        if type(string) is str:
            string = string.encode()
        return self.model.tokenize(string)

    def decode(self, ids, **kwargs):
        detokenized = self.model.detokenize(ids)
        try:
            # Attempt strict UTF-8 decoding first
            return detokenized.decode('utf-8', 'strict')
        except UnicodeDecodeError as e:
            # Log the error and fall back to UTF-8 with replacement
            logger.warning(f"Invalid UTF-8 in detokenized output. Using replacement characters.\n{e}")
            return detokenized.decode('utf-8', 'replace')

    def get_logits(self, tokens):
        self.model.reset()
        self.model.eval(tokens)
        logits = self.model._scores
        logits = np.expand_dims(logits, 0)  # batch dim is expected
        return torch.tensor(logits, dtype=torch.float32)

    def load_grammar(self, string):
        if string != self.grammar_string:
            self.grammar_string = string
            if string.strip() != '':
                self.grammar = llama_cpp_lib(self.loader_config).LlamaGrammar.from_string(string)
            else:
                self.grammar = None

    def generate(self, prompt, state, callback=None, max_prompt_length: int = -1):
        LogitsProcessorList = llama_cpp_lib(self.loader_config).LogitsProcessorList
        prompt = prompt if type(prompt) is str else prompt.decode()

        # Handle truncation
        prompt = self.encode(prompt)
        if max_prompt_length > 0:
            prompt = prompt[-max_prompt_length:]
        prompt = self.decode(prompt)

        self.load_grammar(state['grammar_string'])
        logit_processors = LogitsProcessorList()
        if state['ban_eos_token']:
            logit_processors.append(partial(ban_eos_logits_processor, self.model.token_eos()))

        if state['custom_token_bans']:
            to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
            if len(to_ban) > 0:
                logit_processors.append(partial(custom_token_ban_logits_processor, to_ban))

        completion_chunks = self.model.create_completion(
            prompt=prompt,
            max_tokens=state['max_new_tokens'],
            temperature=state['temperature'],
            top_p=state['top_p'] if state['top_p'] < 1 else 0.999,
            min_p=state['min_p'],
            typical_p=state['typical_p'],
            frequency_penalty=state['frequency_penalty'],
            presence_penalty=state['presence_penalty'],
            repeat_penalty=state['repetition_penalty'],
            top_k=state['top_k'],
            stream=True,
            seed=int(state['seed']) if state['seed'] != -1 else None,
            tfs_z=state['tfs'],
            mirostat_mode=int(state['mirostat_mode']),
            mirostat_tau=state['mirostat_tau'],
            mirostat_eta=state['mirostat_eta'],
            logits_processor=logit_processors,
            grammar=self.grammar
        )

        output = ""
        for completion_chunk in completion_chunks:
            if shared.stop_everything:
                break
            text = completion_chunk['choices'][0]['text']
            output += text
            if callback:
                callback(text)
        return output

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply


def get_llamacpp_cache_type_for_string(quant_type: str):
    quant_type = quant_type.lower()
    if quant_type in llamacpp_valid_cache_types:
        return llamacpp_quant_mapping[quant_type]
    raise ValueError(
        f"Invalid cache type for llama.cpp: {quant_type}. Valid options are: fp16, q8_0, q4_0."
    )


def ban_eos_logits_processor(eos_token, input_ids, logits):
    logits[eos_token] = -float('inf')
    return logits


def custom_token_ban_logits_processor(token_ids, input_ids, logits):
    for token_id in token_ids:
        logits[token_id] = -float('inf')
    return logits
