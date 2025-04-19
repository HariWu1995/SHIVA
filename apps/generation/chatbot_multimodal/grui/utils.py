import re
import gradio as gr

from copy import deepcopy
from pathlib import Path
from datetime import datetime

import torch
from transformers import is_torch_xpu_available

from . import shared as ui
from ..src import shared
from ..src.utils import natural_keys
from ..src.loaders import get_model_metadata
from ..src.sampler import loaders_and_params


def current_time():
    return f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"


symbols = dict(
    refresh = '🔄',
    delete = '🗑️',
    save = '💾',
)


# Helper function to get multiple values from ui.gradio
def gradget(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]
    return [ui.gradio[k] for k in keys if k in ui.gradio.keys()]


def get_fallback_settings():
    return {
        'use_bf16': False,
        'attn_implementation': 'eager',
        'max_seq_len': 2048,
        'n_ctx': 2048,
        'rope_freq_base': 0,
        'compress_pos_emb': 1,
        'alpha_value': 1,
        'truncation_length'     : ui.settings['truncation_length'],
        'truncation_length_info': ui.settings['truncation_length'],
        'skip_special_tokens'   : ui.settings['skip_special_tokens'],
        'custom_stopping_strings': ui.settings['custom_stopping_strings'],
    }


def save_settings(state, preset, extensions_list, show_controls, theme_state,
                settings, default_settings):

    output = deepcopy(settings)
    exclude = ['name2', 'greeting', 'context', 'truncation_length', 'instruction_template_str']
    for k in state:
        if k in settings \
        and k not in exclude:
            output[k] = state[k]

    output['preset'] = preset
    output['character'] = state['character_menu']
    output['prompt-default'] = state['prompt_menu-default']
    output['prompt-notebook'] = state['prompt_menu-notebook']
    output['default_extensions'] = extensions_list
    output['show_controls'] = show_controls
    output['dark_theme'] = (theme_state == 'dark')
    output['seed'] = int(output['seed'])

    # Save extension values in the UI
    for extension_name in extensions_list:
        extension = getattr(extensions, extension_name, None)
        if not extension:
            continue
        extension = extension.script
        params = getattr(extension, 'params', None)
        if not params:
            continue
        for param in params:
            _id = f"{extension_name}-{param}"
            # Only save if different from default value
            if param not in default_settings or \
            params[param] != default_settings[param]:
                output[_id] = params[param]

    # Only keep changed settings
    for key in list(output.keys()):
        if key not in default_settings:
            continue
        if output[key] != default_settings[key]:
            continue
        output.pop(key)

    return yaml.dump(output, sort_keys=False, width=float("inf"), allow_unicode=True)


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_classes, interactive=True):
    """
    Copied from https://github.com/AUTOMATIC1111/stable-diffusion-webui
    """
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args
        return gr.update(**(args or {}))

    refresh_button = gr.Button(value=symbols["refresh"], elem_classes=elem_classes, interactive=interactive)
    refresh_button.click(
        fn=lambda: {k: tuple(v) if type(k) is list else v for k, v in refresh().items()},
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button


def list_model_elements():
    elements = [
        'filter_by_loader',
        'loader',
    ]

    # if is_torch_xpu_available():
    #     for i in range(torch.xpu.device_count()):
    #         elements.append(f'gpu_memory_{i}')
    # else:
    #     for i in range(torch.cuda.device_count()):
    #         elements.append(f'gpu_memory_{i}')
    return elements


def list_interface_input_elements():
    elements = [
        'temperature',
        'temperature_last',
        'dynamic_temperature',
        'dynatemp_low',
        'dynatemp_high',
        'dynatemp_exponent',
        'smoothing_factor',
        'smoothing_curve',
        'min_p',
        'top_p',
        'typical_p',
        'top_k',
        'top_a',
        'top_n_sigma',
        'xtc_threshold',
        'xtc_probability',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'dry_base',
        'dry_multiplier',
        'dry_allowed_length',
        'dry_sequence_breakers',
        'penalty_alpha',
        'frequency_penalty',
        'presence_penalty',
        'encoder_repetition_penalty',
        'repetition_penalty',
        'repetition_penalty_range',
        'no_repeat_ngram_size',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'prompt_lookup_num_tokens',
        'max_new_tokens',
        'max_tokens_second',
        'max_updates_second',
        'auto_max_new_tokens',
        'ban_eos_token',
        'add_bos_token',
        'skip_special_tokens',
        'custom_token_bans',
        'stream',
        'static_cache',
        'truncation_length',
        'seed',
        'guidance_scale',
        'do_sample',
        'sampler_priority',
        'show_after',
        'negative_prompt',
        'grammar_string',
        'custom_stopping_strings',
    ]

    # Chat elements
    elements += [
        'history',
        'search_chat',
        'unique_id',
        'textbox',
        'start_with',
        'mode',
        'chat_style',
        'chat-instruct_command',
        'character_menu',
        'name1',
        'name2',
        'context',
        'greeting',
        'user_bio',
        'custom_system_message',
        'instruction_template_str',
        'chat_template_str',
    ]

    # Notebook/default elements
    elements += [
        'textbox-default',
        'textbox-notebook',
        'prompt_menu-default',
        'prompt_menu-notebook',
        'output_textbox',
    ]

    # Model elements
    elements += list_model_elements()

    return elements


def gather_interface_values(*args):
    interface_elements = list_interface_input_elements()

    output = {}
    for element, value in zip(interface_elements, args):
        output[element] = value

    if not ui.multi_user:
        ui.persistent_interface_state = output

    return output


def apply_interface_values(state, use_persistent=False):
    if use_persistent:
        state = ui.persistent_interface_state
        if 'textbox-default' in state and \
        'prompt_menu-default' in state:
            state.pop('prompt_menu-default')

        if 'textbox-notebook' and \
        'prompt_menu-notebook' in state:
            state.pop('prompt_menu-notebook')

    elements = list_interface_input_elements()

    if len(state) == 0:
        return [gr.update() for k in elements]  # Dummy, do nothing
    else:
        return [state[k] if k in state else gr.update() for k in elements]


def apply_model_settings_to_state(model, state):
    '''
    UI: update the state variable with the model settings
    '''
    model_settings = get_model_metadata(model)

    if 'loader' in model_settings:
        loader = model_settings.pop('loader')

        # If the user is using an alternative loader for the same model type, let them keep using it
        if not (loader == 'ExLlamav2_HF' \
        and state['loader'] in ['ExLlamav2']):
            state['loader'] = loader

    for k in model_settings:
        if k in state:
            state[k] = model_settings[k]

    return state


def save_model_settings(model, state):
    '''
    Save the settings for this model to models/config-user.yaml
    '''
    if model == 'None':
        yield ("Not saving the settings because no model is selected in the menu.")
        return

    user_config = shared.user_config
    model_regex = model + '$'  # For exact matches
    if model_regex not in user_config:
        user_config[model_regex] = {}

    for k in ui.list_model_elements():
        if k == 'loader' \
        or k in loaders_and_params[state['loader']]:
            user_config[model_regex][k] = state[k]

    shared.user_config = user_config
    uconfig = yaml.dump(user_config, sort_keys=False)

    path = str(shared.user_config_path)
    with open(path, 'w') as f:
        f.write(uconfig)

    yield (f"Settings for `{model}` saved to `{path}`.")


def update_model_parameters(state, initial=False):
    '''
    UI: update the command-line arguments based on the interface values
    '''
    elements = list_model_elements()  # the names of the parameters
    gpu_memories = []

    for i, element in enumerate(elements):
        if element not in state:
            continue

        value = state[element]
        if element.startswith('gpu_memory'):
            gpu_memories.append(value)
            continue

        if initial and (element in shared.args_provided):
            continue

        if (value == 0) and (element in ['cpu_memory']):
            value = vars(shared.args_default)[element]

        # Making some simple conversions
        if element == 'cpu_memory' and value is not None:
            value = f"{value}MiB"

        setattr(shared.args, element, value)

    found_positive = False
    for i in gpu_memories:
        if i > 0:
            found_positive = True
            break

    if not (initial and vars(shared.args)['gpu_memory'] != vars(shared.args_default)['gpu_memory']):
        if found_positive:
            shared.args.gpu_memory = [f"{i}MiB" for i in gpu_memories]
        else:
            shared.args.gpu_memory = None


def save_instruction_template(model, template):
    '''
    Similar to the function above, but it saves only the instruction template.
    '''
    if model == 'None':
        yield ("Not saving the template because no model is selected in the menu.")
        return

    user_config = shared.user_config
    model_regex = model + '$'  # For exact matches
    if model_regex not in user_config:
        user_config[model_regex] = {}

    if template == 'None':
        user_config[model_regex].pop('instruction_template', None)
    else:
        user_config[model_regex]['instruction_template'] = template

    shared.user_config = user_config
    uconfig = yaml.dump(user_config, sort_keys=False)

    path = str(shared.user_config_path)
    with open(path, 'w') as f:
        f.write(uconfig)

    if template == 'None':
        yield (f"Instruction template for `{model}` unset in `{path}`, as the value for template was `{template}`.")
    else:
        yield (f"Instruction template for `{model}` saved to `{path}` as `{template}`.")


def get_available_extensions():
    extension_dir = Path(__file__).resolve().parent / 'extensions'
    extensions = sorted(
        set(map(lambda x: x.parts[1], extension_dir.glob('*/script.py'))), key=natural_keys)
    # extensions = [v for v in extensions if v not in github.new_extensions]
    return extensions

