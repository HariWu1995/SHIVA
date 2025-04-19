import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .. import shared
from ..logging import logger
from .llamacpp_model import get_llamacpp_cache_type_for_string
from .llama_cpp_python_hijack import llama_cpp_lib


class LlamacppHF(PreTrainedModel):

    def __init__(self, model, path, loader_config):
        super().__init__(PretrainedConfig())
        self.model = model
        self.loader_config = loader_config
        self.generation_config = GenerationConfig()

        self.past_seq = None
        self.llamacpp_cache = {
            'n_tokens': self.model.n_tokens,
            'input_ids': self.model.input_ids,
            'scores': self.model.scores,
            'ctx': self.model._ctx.ctx
        }

        if shared.args.cfg_cache:
            self.past_seq_negative = None
            self.llamacpp_cache_negative = {
                'n_tokens': self.model.n_tokens,
                'input_ids': self.model.input_ids.copy(),
                'scores': self.model.scores.copy(),
                'ctx': llama_cpp_lib(self.loader_config).llama_new_context_with_model(model.model, model.context_params)
            }

    def _validate_model_class(self):
        pass

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_ids': input_ids, **kwargs}

    def save_cache(self):
        self.llamacpp_cache.update({
            'n_tokens': self.model.n_tokens,
            'input_ids': self.model.input_ids,
            'scores': self.model.scores,
            'ctx': self.model._ctx.ctx
        })

    def save_negative_cache(self):
        self.llamacpp_cache_negative.update({
            'n_tokens': self.model.n_tokens,
            'input_ids': self.model.input_ids,
            'scores': self.model.scores,
            'ctx': self.model._ctx.ctx
        })

    def load_cache(self):
        self.model.n_tokens = self.llamacpp_cache['n_tokens']
        self.model.input_ids = self.llamacpp_cache['input_ids']
        self.model.scores = self.llamacpp_cache['scores']
        self.model._ctx.ctx = self.llamacpp_cache['ctx']

    def load_negative_cache(self):
        self.model.n_tokens = self.llamacpp_cache_negative['n_tokens']
        self.model.input_ids = self.llamacpp_cache_negative['input_ids']
        self.model.scores = self.llamacpp_cache_negative['scores']
        self.model._ctx.ctx = self.llamacpp_cache_negative['ctx']

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    def __call__(self, *args, **kwargs):
        use_cache = kwargs.get('use_cache', True)
        labels = kwargs.get('labels', None)
        past_key_values = kwargs.get('past_key_values', None)

        if len(args) > 0:
            if not self.loader_config.cfg_cache:
                logger.error("Please enable the cfg-cache option to use CFG with llamacpp_HF.")
                return

            input_ids = args[0]
            is_negative = True
            past_seq = self.past_seq_negative
            self.load_negative_cache()
        else:
            input_ids = kwargs['input_ids']
            is_negative = False
            past_seq = self.past_seq
            self.load_cache()

        seq = input_ids[0].tolist()
        if is_negative and past_key_values is not None:
            seq = past_key_values + seq

        seq_tensor = torch.tensor(seq)
        reset = True

        # Make the forward call. The prefix-match code has been adapted from
        # https://github.com/abetlen/llama-cpp-python/commit/f4090a0bb2a2a25acfe28d31c82cc1aa273bedee
        if labels is None:
            if past_seq is not None:
                min_length = min(past_seq.shape[0], seq_tensor.shape[0])
                indices = torch.nonzero(~torch.eq(past_seq[:min_length], seq_tensor[:min_length]))
                if len(indices) > 0:
                    longest_prefix = indices[0].item()
                else:
                    longest_prefix = min_length

                if longest_prefix > 0:
                    reset = False
                    self.model.n_tokens = longest_prefix
                    if len(seq_tensor) - longest_prefix > 0:
                        self.model.eval(seq[longest_prefix:])
                    else:
                        self.model.n_tokens -= 1
                        self.model.eval([seq[-1]])

            if reset:
                self.model.reset()
                self.model.eval(seq)

            logits = torch.tensor(self.model.scores[self.model.last_updated_index, :]).view(1, 1, -1).to(input_ids.device)
        else:
            self.model.reset()
            self.model.eval(seq)
            logits = torch.tensor(self.model.eval_logits)
            logits = logits.view(1, logits.shape[0], logits.shape[1]).to(input_ids.device)

        if is_negative:
            self.save_negative_cache()
            self.past_seq_negative = seq_tensor
        else:
            self.save_cache()
            self.past_seq = seq_tensor

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(logits=logits, past_key_values=seq if use_cache else None, loss=loss)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        assert len(model_args) == 0 and len(kwargs) == 0, "extra args is currently not supported"

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        if pretrained_model_name_or_path.is_file():
            model_file = pretrained_model_name_or_path
        else:
            model_file = sorted(pretrained_model_name_or_path.glob('*.gguf'))[0]

        logger.info(f"llama.cpp weights detected: {model_file}\n")

        if self.loader_config.tensor_split is None \
        or self.loader_config.tensor_split.strip() == '':
            tensor_split_list = None
        else:
            tensor_split_list = [float(x) for x in self.loader_config.tensor_split.strip().split(",")]

        params = {
            'model_path': str(model_file),
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
            'logits_all': self.loader_config.logits_all,
        }

        if shared.args.cache_type != 'fp16':
            params["type_k"] = get_llamacpp_cache_type_for_string(shared.args.cache_type)
            params["type_v"] = get_llamacpp_cache_type_for_string(shared.args.cache_type)

        Llama = llama_cpp_lib(self.loader_config).Llama
        try:
            model = Llama(**params)
        except Exception as e:
            error_message = (
                f"Failed loading the model. **This usually happens due to lack of memory**. Try these steps:\n"
                f"1. Reduce the context length `n_ctx` (currently {self.loader_config.n_ctx})."
                f"{' Try a lower value like 4096.' if self.loader_config.n_ctx > 4096 else '.'}"
                "\n"
                f"2. Lower the `n-gpu-layers` value (currently {self.loader_config.n_gpu_layers})."
            )
            raise type(e)(error_message) from e

        model.last_updated_index = -1
        return LlamacppHF(model, model_file)
