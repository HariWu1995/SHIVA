import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from exllamav3 import Cache, Config, Model

from .. import shared
from ..logging import logger

try:
    import flash_attn
except Exception:
    logger.warning('Failed to load flash-attention due to the following error:\n')
    traceback.print_exc()


class Exllamav3HF(PreTrainedModel):

    def __init__(self, model_dir, loader_config):
        super().__init__(PretrainedConfig())
        self.generation_config = GenerationConfig()
        self.loader_config = loader_config

        config = Config.from_directory(model_dir)
        self.ex_model = Model.from_config(config)

        # Calculate the closest multiple of 256 at or above the chosen value
        max_tokens = self.loader_config.max_seq_len
        if max_tokens % 256 != 0:
            adjusted_tokens = ((max_tokens // 256) + 1) * 256
            logger.warning(f"max_num_tokens must be a multiple of 256. Adjusting from {max_tokens} to {adjusted_tokens}")
            max_tokens = adjusted_tokens

        self.ex_cache = Cache(self.ex_model, max_num_tokens=max_tokens)

        # Create load parameters dictionary
        load_params = {'progressbar': True}
        if self.loader_config.gpu_split:
            split = [float(alloc) for alloc in self.loader_config.gpu_split.split(",")]
            load_params['use_per_device'] = split

        self.ex_model.load(**load_params)
        self.past_seq = None
        self.max_tokens = max_tokens

    def _validate_model_class(self):
        pass

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_ids': input_ids, **kwargs}

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    def __call__(self, *args, **kwargs):

        labels = kwargs.get('labels', None)
        use_cache = kwargs.get('use_cache', True)
        past_key_values = kwargs.get('past_key_values', None)

        if len(args) > 0:
            if not self.loader_config.cfg_cache:
                logger.error("Please enable the cfg-cache option to use CFG with ExLlamav3_HF.")
                return

            input_ids = args[0]
            is_negative = True
            past_seq = self.past_seq_negative
            ex_cache = self.ex_cache_negative

        else:
            input_ids = kwargs['input_ids']
            is_negative = False
            past_seq = self.past_seq
            ex_cache = self.ex_cache

        seq = input_ids[0].tolist()
        if is_negative and past_key_values is not None:
            seq = past_key_values + seq

        tensor_seq = torch.tensor(seq)
        reset = True

        # Make the forward call
        if labels is None:

            if past_seq is not None:
                min_length = min(past_seq.shape[0], tensor_seq.shape[0])
                indices = torch.nonzero(~torch.eq(past_seq[:min_length], 
                                                tensor_seq[:min_length]))
                if len(indices) > 0:
                    longest_prefix = indices[0].item()
                else:
                    longest_prefix = min_length

                if longest_prefix > 0:
                    reset = False
                    current_len = longest_prefix
                    if len(tensor_seq) - longest_prefix > 1:
                        self.ex_model.forward(
                            input_ids=tensor_seq[longest_prefix:-1].view(1, -1),
                            params={
                                "attn_mode": "flash_attn",
                                "cache": ex_cache,
                                "past_len": longest_prefix,
                                "batch_shape": (1, self.max_tokens)
                            }
                        )
                        current_len = longest_prefix + len(tensor_seq) - longest_prefix - 1

            if reset:
                if len(tensor_seq) > 1:
                    self.ex_model.forward(
                        input_ids=tensor_seq[:-1].view(1, -1),
                        params={
                            "attn_mode": "flash_attn",
                            "cache": ex_cache,
                            "past_len": 0,
                            "batch_shape": (1, self.max_tokens)
                        }
                    )
                    current_len = len(tensor_seq) - 1
                else:
                    current_len = 0

            logits = self.ex_model.forward(
                input_ids=tensor_seq[-1:].view(1, -1),
                params={
                    "attn_mode": "flash_attn",
                    "cache": ex_cache,
                    "past_len": current_len,
                    "batch_shape": (1, self.max_tokens)
                }
            ).to(input_ids.device).float()

        else:
            logits = self.ex_model.forward(
                input_ids=tensor_seq.view(1, -1),
                params={
                    "attn_mode": "flash_attn",
                    "cache": ex_cache,
                    "past_len": 0,
                    "batch_shape": (1, self.max_tokens)
                }
            ).float()

        if is_negative:
            self.past_seq_negative = tensor_seq
        else:
            self.past_seq = tensor_seq

        loss = None
        if labels is not None:

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:    ].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, logits.shape[-1])
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(logits=logits, past_key_values=seq if use_cache else None, loss=loss)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], 
        loader_config,
        *model_args, 
        **kwargs
    ):
        # assert len(model_args) == 0 \
        #     and len(kwargs) == 0, "extra args is currently not supported"
        
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        return Exllamav3HF(pretrained_model_name_or_path, loader_config)
