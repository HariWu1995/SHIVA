import time
import re

from .. import shared, gguf
from ..logging import logger
from ..instruct import load_instruction_template
from ..models import clear_model_cache, last_generation_time

from .huggingface import huggingface_loader
from .llama_cpp import llamacpp_HF_loader, llamacpp_loader
from .ex_llama import ExLlamav2_loader, ExLlamav2_HF_loader, ExLlamav3_HF_loader
from .tensorrt import TensorRT_LLM_loader


load_func_map = {
    'Transformers': huggingface_loader,
    'llama.cpp'   : llamacpp_loader,
    'llamacpp_HF' : llamacpp_HF_loader,
    'ExLlamav3_HF': ExLlamav3_HF_loader,
    'ExLlamav2_HF': ExLlamav2_HF_loader,
    'ExLlamav2'   : ExLlamav2_loader,
    'TensorRT-LLM': TensorRT_LLM_loader,
}


def load_model(model_name, loader=None):

    shared.is_seq2seq = False
    shared.model_name = model_name

    metadata = get_model_metadata(model_name)

    logger.info(f"Loading \"{model_name}\"")
    t0 = time.time()

    if loader is None:
        if shared.loader is not None:
            loader = shared.loader
        else:
            loader = metadata['loader']
    if loader is None:
        logger.error('The path to the model does not exist. Exiting.')
        raise ValueError

    shared.loader = loader
    clear_model_cache()
    output = load_func_map[loader](model_name)
    if type(output) is tuple:
        model, tokenizer = output
    else:
        model = output
        if model is None:
            return None, None
        else:
            tokenizer = load_tokenizer(model_name)

    # shared.settings.update({k: v for k, v in metadata.items() if k in shared.settings})
    # if loader.lower().startswith('exllama') \
    # or loader.lower().startswith('tensorrt'):
    #     shared.settings['truncation_length'] = shared.args.max_seq_len
    # elif loader in ['llama.cpp', 'llamacpp_HF']:
    #     shared.settings['truncation_length'] = shared.args.n_ctx

    logger.info(f"Loaded \"{model_name}\" in {(time.time()-t0):.2f} seconds.")
    logger.info(f"LOADER: \"{loader}\"")
    # logger.info(f"TRUNCATION LENGTH: {shared.settings['truncation_length']}")
    logger.info(f"INSTRUCTION TEMPLATE: \"{metadata['instruction_template']}\"")
    return model, tokenizer


def unload_model(keep_model_name=False):
    shared.model = None
    shared.tokenizer = None
    shared.lora_names = []

    clear_model_cache()
    if not keep_model_name:
        shared.model_name = 'None'


def reload_model():
    unload_model()
    shared.model, shared.tokenizer = load_model(shared.model_name)


def unload_model_if_idle():
    global last_generation_time

    logger.info(f"Setting a timeout of {shared.args.idle_timeout} minutes to unload the model in case of inactivity.")

    while True:
        shared.generation_lock.acquire()
        try:
            if time.time() - last_generation_time > shared.args.idle_timeout * 60:
                if shared.model is not None:
                    logger.info("Unloading the model for inactivity.")
                    unload_model(keep_model_name=True)
        finally:
            shared.generation_lock.release()

        time.sleep(60)


def get_model_metadata(model):
    
    assert model_name in shared.MLM_LOCAL_MODELS.keys()
    path_to_model = Path(shared.MLM_LOCAL_MODELS[model_name])

    model_settings = {}

    # Get settings from models/config.yaml and models/config-user.yaml
    default_settings = shared.model_config
    for pat in default_settings:
        if re.match(pat.lower(), model.lower()):
            for k in default_settings[pat]:
                model_settings[k] = default_settings[pat][k]

    config_path = path_to_model / 'config.json'
    if config_path.exists():
        hf_metadata = json.loads(open(config_path, 'r', encoding='utf-8').read())
    else:
        hf_metadata = None

    if 'loader' not in model_settings:
        model_settings['loader'] = infer_loader(model, model_settings, path_to_model)

    # GGUF metadata
    if model_settings['loader'] in ['llama.cpp', 'llamacpp_HF']:

        if path_to_model.is_file():
            model_file = path_to_model
        else:
            model_file = list(path_to_model.glob('*.gguf'))[0]
        metadata = gguf.load_metadata(model_file)

        for k in metadata:
            if k.endswith('context_length'):
                model_settings['n_ctx'] = min(metadata[k], 8192)
                model_settings['truncation_length_info'] = metadata[k]
            elif k.endswith('rope.freq_base'):
                model_settings['rope_freq_base'] = metadata[k]
            elif k.endswith('rope.scale_linear'):
                model_settings['compress_pos_emb'] = metadata[k]
            elif k.endswith('rope.scaling.factor'):
                model_settings['compress_pos_emb'] = metadata[k]
            elif k.endswith('block_count'):
                model_settings['n_gpu_layers'] = metadata[k] + 1

        if 'tokenizer.chat_template' in metadata:
            template = metadata['tokenizer.chat_template']
            eos_token = metadata['tokenizer.ggml.tokens'][metadata['tokenizer.ggml.eos_token_id']]
            if 'tokenizer.ggml.bos_token_id' in metadata:
                bos_token = metadata['tokenizer.ggml.tokens'][metadata['tokenizer.ggml.bos_token_id']]
            else:
                bos_token = ""

            template = template.replace('eos_token', "'{}'".format(eos_token))
            template = template.replace('bos_token', "'{}'".format(bos_token))

            template = re.sub(r'raise_exception\([^)]*\)', "''", template)
            template = re.sub(r'{% if add_generation_prompt %}.*', '', template, flags=re.DOTALL)
            model_settings['instruction_template'] = 'Custom (obtained from model metadata)'
            model_settings['instruction_template_str'] = template

    else:
        # Transformers metadata
        if hf_metadata is not None:
            metadata = json.loads(open(config_path, 'r', encoding='utf-8').read())
            if 'pretrained_config' in metadata:
                metadata = metadata['pretrained_config']

            for k in ['max_position_embeddings', 'model_max_length', 'max_seq_len']:
                if k in metadata:
                    model_settings['truncation_length'] = metadata[k]
                    model_settings['truncation_length_info'] = metadata[k]
                    model_settings['max_seq_len'] = min(metadata[k], 8192)

            if 'rope_theta' in metadata:
                model_settings['rope_freq_base'] = metadata['rope_theta']
            elif 'attn_config' in metadata \
              and 'rope_theta' in metadata['attn_config']:
                model_settings['rope_freq_base'] = metadata['attn_config']['rope_theta']

            if 'rope_scaling' in metadata \
                and isinstance(metadata['rope_scaling'], dict) \
                and all(key in metadata['rope_scaling'] for key in ('type', 'factor')):
                if metadata['rope_scaling']['type'] == 'linear':
                    model_settings['compress_pos_emb'] = metadata['rope_scaling']['factor']

            # For Gemma-2
            if 'torch_dtype' in metadata \
            and metadata['torch_dtype'] == 'bfloat16':
                model_settings['bf16'] = True

            # For Gemma-2
            model_arch = 'Gemma2ForCausalLM'
            if 'architectures' in metadata \
                   and isinstance(metadata['architectures'], list) \
                and model_arch in metadata['architectures']:
                model_settings['use_eager_attention'] = True

    # Try to find the Jinja instruct template
    config_path = path_to_model / 'tokenizer_config.json'
    if config_path.exists():
        metadata = json.loads(open(config_path, 'r', encoding='utf-8').read())
        if 'chat_template' in metadata:
            template = metadata['chat_template']
            if isinstance(template, list):
                template = template[0]['template']

            for k in ['eos_token', 'bos_token']:
                if k in metadata:
                    value = metadata[k]
                    if isinstance(value, dict):
                        value = value['content']
                    template = template.replace(k, "'{}'".format(value))

            template = re.sub(r'raise_exception\([^)]*\)', "''", template)
            template = re.sub(r'{% if add_generation_prompt %}.*', '', template, flags=re.DOTALL)

            model_settings['instruction_template'] = 'Custom (obtained from model metadata)'
            model_settings['instruction_template_str'] = template

    if 'instruction_template' not in model_settings:
        model_settings['instruction_template'] = 'Alpaca'

    # Ignore rope_freq_base if set to the default value
    if 'rope_freq_base' in model_settings \
    and model_settings['rope_freq_base'] == 10_000:
        model_settings.pop('rope_freq_base')

    # Apply user settings from models/config-user.yaml
    settings = shared.user_config
    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    # Load instruction template if defined by name rather than by value
    if model_settings['instruction_template'] != 'Custom (obtained from model metadata)':
        model_settings['instruction_template_str'] = load_instruction_template(model_settings['instruction_template'])
    return model_settings


def infer_loader(model_name, model_settings, path_to_model):
    model_name = model_name.lower()
    if not path_to_model.exists():
        return None
    elif len(list(path_to_model.glob('*.gguf'))) > 0 \
              and path_to_model.is_dir() \
              and (path_to_model / 'tokenizer_config.json').exists():
        return 'llamacpp_HF'
    elif len(list(path_to_model.glob('*.gguf'))) > 0:
        return 'llama.cpp'
    elif re.match(r'.*\.gguf', model_name):
        return 'llama.cpp'
    elif re.match(r'.*exl3', model_name):
        return 'ExLlamav3_HF'
    elif re.match(r'.*exl2', model_name):
        return 'ExLlamav2_HF'
    elif re.match(r'.*-hqq', model_name):
        return 'HQQ'
    return 'Transformers'

