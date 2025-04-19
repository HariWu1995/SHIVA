from pathlib import Path, PurePath

import torch
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner, ModelRunnerCpp

from .. import shared
from ..logging import logger
from modules.text_generation import (
    get_max_prompt_length,
    get_reply_from_output_ids
)


class TensorRTLLModel:

    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path_to_model, loader_config):

        if isinstance(path_to_model, PurePath):
            path_to_model = str(path_to_model)

        runtime_rank = tensorrt_llm.mpi_rank()

        # Define model settings
        runner_kwargs = dict(
            engine_dir=path_to_model,
            rank=runtime_rank,
            lora_dir=None,
            debug_mode=False,
            lora_ckpt_source="hf",
        )

        if shared.args.cpp_runner:
            logger.info("TensorRT-LLM: Using \"ModelRunnerCpp\"")
            runner_kwargs.update(
                max_batch_size=1,
                max_input_len=loader_config.max_seq_len - 512,
                max_output_len=512,
                max_beam_width=1,
                max_attention_window_size=None,
                sink_token_length=None,
            )
        else:
            logger.info("TensorRT-LLM: Using \"ModelRunner\"")

        # Load the model
        runner_cls = ModelRunnerCpp if shared.args.cpp_runner else ModelRunner
        runner = runner_cls.from_dir(**runner_kwargs)

        result = self()
        result.model = runner
        result.runtime_rank = runtime_rank
        result.loader_config = loader_config

        return result

    def generate_with_streaming(self, prompt, state, max_prompt_length: int = 0):
        batch_input_ids = []
        input_ids = shared.tokenizer.encode(prompt, add_special_tokens=True, truncation=False)
        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        input_ids = input_ids[-max_prompt_length:]  # Apply truncation_length
        batch_input_ids.append(input_ids)

        if self.loader_config.cpp_runner:
            max_new_tokens = min(512, state['max_new_tokens'])
        elif state['auto_max_new_tokens']:
            max_new_tokens = state['truncation_length'] - input_ids.shape[-1]
        else:
            max_new_tokens = state['max_new_tokens']

        with torch.no_grad():
            generator = self.model.generate(
                batch_input_ids,
                max_new_tokens=max_new_tokens,
                max_attention_window_size=None,
                sink_token_length=None,
                end_id=shared.tokenizer.eos_token_id if not state['ban_eos_token'] else -1,
                pad_id=shared.tokenizer.pad_token_id or shared.tokenizer.eos_token_id,
                temperature=state['temperature'],
                top_k=state['top_k'],
                top_p=state['top_p'],
                num_beams=1,
                length_penalty=1.0,
                repetition_penalty=state['repetition_penalty'],
                presence_penalty=state['presence_penalty'],
                frequency_penalty=state['frequency_penalty'],
                stop_words_list=None,
                bad_words_list=None,
                lora_uids=None,
                prompt_table_path=None,
                prompt_tasks=None,
                streaming=not self.loader_config.cpp_runner,
                output_sequence_lengths=True,
                return_dict=True,
                medusa_choices=None
            )

        torch.cuda.synchronize()

        cumulative_reply = ''
        starting_from = batch_input_ids[0].shape[-1]

        if self.loader_config.cpp_runner:
            seq_length = generator['sequence_lengths'][0].item()
            output_ids = generator['output_ids'][0][0][:seq_length].tolist()

            cumulative_reply += get_reply_from_output_ids(output_ids, state, starting_from=starting_from)
            starting_from = seq_length
            yield cumulative_reply

        else:
            for curr_outputs in generator:
                if shared.stop_everything:
                    break

                seq_length = curr_outputs['sequence_lengths'][0].item()
                output_ids = curr_outputs['output_ids'][0][0][:seq_length].tolist()

                cumulative_reply += get_reply_from_output_ids(output_ids, state, starting_from=starting_from)
                starting_from = seq_length
                yield cumulative_reply

    def generate(self, prompt, state):
        output = ''
        for output in self.generate_with_streaming(prompt, state):
            pass
        return output
