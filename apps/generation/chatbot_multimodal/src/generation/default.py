import time
import traceback
import transformers
from transformers import LogitsProcessorList, StoppingCriteriaList

from .. import shared
from ..logging import logger, pprinter
from ..models import clear_model_cache
from ..callbacks import Iteratorize, Stream, _StopEverythingStoppingCriteria
from ..grammar.utils import initialize_grammar
from ..grammar.logits import GrammarConstrainedLogitsProcessor

from .utils import set_manual_seed, print_prompt


def generate_reply_HF(question, original_question, seed, state, stopping_strings=None, is_chat=False):

    if shared.loader == 'Transformers':
        clear_model_cache()

    generate_params = {}
    important_params = [
        'temperature', 'temperature_last',
        'dynamic_temperature', 'dynatemp_low', 'dynatemp_high', 'dynatemp_exponent',
        'smoothing_factor', 'smoothing_curve',
        'min_p', 'top_p', 'typical_p',
        'top_a', 'top_n_sigma',
        'top_k', 'guidance_scale', 'max_new_tokens', 'do_sample',
        'xtc_threshold', 'xtc_probability',
        'tfs',
        'dry_multiplier', 'dry_allowed_length', 'dry_base', 'dry_sequence_breakers',
        'penalty_alpha', 'frequency_penalty', 'presence_penalty', 
        'repetition_penalty', 'repetition_penalty_range',
        'no_repeat_ngram_size', 'encoder_repetition_penalty',
        'mirostat_mode', 'mirostat_tau', 'mirostat_eta',
    ]
    for k in important_params:
        if k in state:
            generate_params[k] = state[k]

    for k in ['epsilon_cutoff', 'eta_cutoff']:
        if state[k] > 0:
            generate_params[k] = state[k] * 1e-4

    if state['prompt_lookup_num_tokens'] > 0:
        generate_params['prompt_lookup_num_tokens'] = state['prompt_lookup_num_tokens']

    if state['ban_eos_token']:
        generate_params['suppress_tokens'] = [shared.tokenizer.eos_token_id]

    if state['static_cache']:
        generate_params['cache_implementation'] = 'static'

    if isinstance(state['sampler_priority'], list) \
          and len(state['sampler_priority']) > 0:
        generate_params['sampler_priority'] = state['sampler_priority']
    elif isinstance(state['sampler_priority'], str) \
                and state['sampler_priority'].strip() != '':
        generate_params['sampler_priority'] = []
        for x in state['sampler_priority'].replace('\n', ',').split(','):
            if x.strip():
                generate_params['sampler_priority'].append(x.strip())

    if state['custom_token_bans']:
        to_ban = [int(x) 
                  for x in state['custom_token_bans'].split(',')]
        if len(to_ban) > 0:
            if generate_params.get('suppress_tokens', None):
                generate_params['suppress_tokens'] += to_ban
            else:
                generate_params['suppress_tokens'] = to_ban

    if state['negative_prompt'] != '':
        generate_params['negative_prompt_ids'] = encode(state['negative_prompt'])

    generate_params.update({'use_cache': not shared.model_loader_config.no_cache})
    if shared.model_loader_config.use_deepspeed:
        generate_params.update({'synced_gpus': True})

    # Encode the input
    input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    output = input_ids[0]
    if state['auto_max_new_tokens']:
        generate_params['max_new_tokens'] = state['truncation_length'] - input_ids.shape[-1]

    # Add the encoded tokens to generate_params
    inputs_embeds = None
    original_input_ids = input_ids
    generate_params.update({'inputs': input_ids})
    if inputs_embeds is not None:
        generate_params.update({'inputs_embeds': inputs_embeds})

    # Stopping criteria / eos token
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    generate_params['eos_token_id'] = eos_token_ids
    generate_params['stopping_criteria'] = StoppingCriteriaList()
    generate_params['stopping_criteria'].append(_StopEverythingStoppingCriteria())

    # Logits processor
    processor = state.get('logits_processor', LogitsProcessorList([]))
    if not isinstance(processor, LogitsProcessorList):
        processor = LogitsProcessorList([processor])

    # Grammar
    if state['grammar_string'].strip() != '':
        grammar = initialize_grammar(state['grammar_string'])
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
        processor.append(grammar_processor)

    generate_params['logits_processor'] = processor

    if shared.args.verbose:
        logger.info("GENERATE_PARAMS=")
        filtered_params = {
                key: value 
            for key, value in generate_params.items() 
            if not isinstance(value, torch.Tensor)
        }
        pprinter.pprint(filtered_params)

        logger.info("PROMPT=")
        print_prompt(decode(input_ids[0], skip_special_tokens=False))

    # Handle StreamingLLM for llamacpp_HF
    if shared.model.__class__.__name__ == 'LlamacppHF' \
    and shared.model_loader_config.streaming_llm:
        tmp = process_llamacpp_cache(shared.model.model, input_ids[-1].tolist(), 
                                     shared.model.model._input_ids.tolist())
        shared.model.past_seq = torch.tensor(tmp)
        shared.model.save_cache()

    t0 = time.time()
    try:
        if not is_chat and not shared.is_seq2seq:
            yield ''

        # Generate the entire reply at once.
        if not state['stream']:
            with torch.no_grad():
                output = shared.model.generate(**generate_params)[0]
                device = shared.get_device()
                if device:
                    output = output.to(device)

            starting_from = 0 if shared.is_seq2seq else len(input_ids[0])
            yield get_reply_from_output_ids(output, state, starting_from=starting_from)

        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator.
        else:

            def generate_with_callback(callback=None, *args, **kwargs):
                kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                with torch.no_grad():
                    shared.model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, [], kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                cumulative_reply = ''
                starting_from = 0 if shared.is_seq2seq else len(input_ids[0])

                for output in generator:
                    if output[-1] in eos_token_ids:
                        break

                    new_content = get_reply_from_output_ids(output, state, starting_from=starting_from)
                    # check the partial unicode character
                    if chr(0xfffd) in new_content:
                        continue

                    cumulative_reply += new_content
                    starting_from = len(output)
                    yield cumulative_reply

    except Exception:
        traceback.print_exc()

    finally:
        t1 = time.time()
        original_tokens = len(original_input_ids[0])
        new_tokens = len(output) - (original_tokens if not shared.is_seq2seq else 0)
        logger.info(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return


def get_max_prompt_length(state):
    return state['truncation_length'] - state['max_new_tokens']


def get_reply_from_output_ids(output_ids, state=None, starting_from=0):
    reply = decode(output_ids[starting_from:], state['skip_special_tokens'] if state else True)

    if reply.startswith(' '):
        return reply

    if not (hasattr(shared.tokenizer, 'convert_ids_to_tokens') and len(output_ids) > starting_from):
        return reply

    # Handle tokenizers that do not add the leading space for the first token
    first_token = shared.tokenizer.convert_ids_to_tokens(int(output_ids[starting_from]))
    if isinstance(first_token, (bytes,)):
        # try to decode the bytes to a string
        # if it fails, which means it's not a string in this turn, just ignore it
        try:
            first_token = first_token.decode('utf8')
        except UnicodeDecodeError:
            first_token = ''

    if first_token.startswith('▁'):
        reply = ' ' + reply
    return reply


def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    if shared.tokenizer is None:
        raise ValueError('No tokenizer is loaded')

    if shared.model.__class__.__name__ in ['LlamaCppModel', 'Exllamav2Model', 'TensorRTLLMModel']:
        input_ids = shared.tokenizer.encode(str(prompt))
        if shared.model.__class__.__name__ not in ['Exllamav2Model']:
            input_ids = np.array(input_ids).reshape(1, len(input_ids))
    else:
        input_ids = shared.tokenizer.encode(str(prompt), return_tensors='pt', add_special_tokens=add_special_tokens)

        if hasattr(shared.tokenizer, 'bos_token_id') \
                and shared.tokenizer.bos_token_id is not None:
            if add_bos_token:
                if (len(input_ids[0]) > 0 and input_ids[0][0] != shared.tokenizer.bos_token_id) \
                or (len(input_ids[0]) == 0):
                    # Add a missing bos token (it may not have been added due to faulty model metadata)
                    bos_tensor = torch.tensor([[shared.tokenizer.bos_token_id]])
                    input_ids = torch.cat((bos_tensor, input_ids), 1)

                # Prevent double bos token due to jinja templates with <s> somewhere
                while len(input_ids[0]) > 1 \
                      and input_ids[0][0] == shared.tokenizer.bos_token_id \
                      and input_ids[0][1] == shared.tokenizer.bos_token_id:
                    input_ids = input_ids[:, 1:]
            else:
                # Remove any bos token that may have been added
                while len(input_ids[0]) > 0 \
                      and input_ids[0][0] == shared.tokenizer.bos_token_id:
                    input_ids = input_ids[:, 1:]

    # Handling truncation
    if truncation_length is not None:
        input_ids = input_ids[:, -truncation_length:]

    if shared.model.__class__.__name__ in ['LlamaCppModel', 'Exllamav2Model', 'TensorRTLLMModel'] \
    or shared.args.cpu:
        return input_ids
    else:
        device = shared.get_device()
        if device:
            return input_ids.to(device)
        return input_ids


def decode(output_ids, skip_special_tokens=True):
    if shared.tokenizer is None:
        raise ValueError('No tokenizer is loaded')
    return shared.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)

