from pathlib import Path

import gradio as gr

from . import shared as ui
from .utils import gradget, create_refresh_button, gather_interface_values

from ..src import shared, preset
from ..src.utils import get_available_presets, get_available_grammars
from ..src.sampler import loaders_and_params, loaders_samplers, list_all_samplers


def create_ui(default_preset):

    is_single_user = not ui.multi_user

    all_loaders = list(loaders_and_params.keys())
    all_grammars = get_available_grammars()
    all_presets = get_available_presets()
    generate_params = preset.load_preset(default_preset)

    with gr.Tab("Parameters", elem_id="parameters"):

        with gr.Tab("Generation"):

            with gr.Row():

                with gr.Column():
                    with gr.Row():
                        ui.gradio['preset_menu'] = gr.Dropdown(choices=all_presets, 
                                                                value=default_preset, label='Preset', elem_classes='slim-dropdown')
                        create_refresh_button(
                            ui.gradio['preset_menu'], 
                            lambda: None, 
                            lambda: {'choices': all_presets}, elem_classes='refresh-button', interactive=is_single_user)
                        
                        ui.gradio['save_preset'] = gr.Button('💾', elem_classes='refresh-button', interactive=is_single_user)
                        ui.gradio['delete_preset'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=is_single_user)
                        ui.gradio['random_preset'] = gr.Button('🎲', elem_classes='refresh-button')

                with gr.Column():
                    ui.gradio['filter_by_loader'] = gr.Dropdown(
                        label="Filter by loader", 
                        choices=["All"] + all_loaders, value="All", elem_classes='slim-dropdown')

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('## Curve shape')
                            ui.gradio['temperature'] = gr.Slider(0.01, 5, value=generate_params['temperature'], step=0.01, label='temperature')
                            ui.gradio['dynatemp_low'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_low'], step=0.01, label='dynatemp_low', visible=generate_params['dynamic_temperature'])
                            ui.gradio['dynatemp_high'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_high'], step=0.01, label='dynatemp_high', visible=generate_params['dynamic_temperature'])
                            ui.gradio['dynatemp_exponent'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_exponent'], step=0.01, label='dynatemp_exponent', visible=generate_params['dynamic_temperature'])
                            ui.gradio['smoothing_factor'] = gr.Slider(0.0, 10.0, value=generate_params['smoothing_factor'], step=0.01, label='smoothing_factor', info='Activates Quadratic Sampling.')
                            ui.gradio['smoothing_curve'] = gr.Slider(1.0, 10.0, value=generate_params['smoothing_curve'], step=0.01, label='smoothing_curve', info='Adjusts the dropoff curve of Quadratic Sampling.')

                            gr.Markdown('## Curve cutoff')
                            ui.gradio['min_p'] = gr.Slider(0.0, 1.0, value=generate_params['min_p'], step=0.01, label='min_p')
                            ui.gradio['top_n_sigma'] = gr.Slider(0.0, 5.0, value=generate_params['top_n_sigma'], step=0.01, label='top_n_sigma')
                            ui.gradio['top_p'] = gr.Slider(0.0, 1.0, value=generate_params['top_p'], step=0.01, label='top_p')
                            ui.gradio['top_k'] = gr.Slider(0, 200, value=generate_params['top_k'], step=1, label='top_k')
                            ui.gradio['typical_p'] = gr.Slider(0.0, 1.0, value=generate_params['typical_p'], step=0.01, label='typical_p')
                            ui.gradio['xtc_threshold'] = gr.Slider(0, 0.5, value=generate_params['xtc_threshold'], step=0.01, label='xtc_threshold', info=ui.element_description['params_xtc_threshold'])
                            ui.gradio['xtc_probability'] = gr.Slider(0, 1, value=generate_params['xtc_probability'], step=0.01, label='xtc_probability', info=ui.element_description['params_xtc_probability'])
                            ui.gradio['epsilon_cutoff'] = gr.Slider(0, 9, value=generate_params['epsilon_cutoff'], step=0.01, label='epsilon_cutoff')
                            ui.gradio['eta_cutoff'] = gr.Slider(0, 20, value=generate_params['eta_cutoff'], step=0.01, label='eta_cutoff')
                            ui.gradio['tfs'] = gr.Slider(0.0, 1.0, value=generate_params['tfs'], step=0.01, label='tfs')
                            ui.gradio['top_a'] = gr.Slider(0.0, 1.0, value=generate_params['top_a'], step=0.01, label='top_a')

                            gr.Markdown('## Repetition suppression')
                            ui.gradio['dry_multiplier'] = gr.Slider(0, 5, value=generate_params['dry_multiplier'], step=0.01, label='dry_multiplier', info='Set to greater than 0 to enable DRY. Recommended value: 0.8.')
                            ui.gradio['dry_allowed_length'] = gr.Slider(1, 20, value=generate_params['dry_allowed_length'], step=1, label='dry_allowed_length', info='Longest sequence that can be repeated without being penalized.')
                            ui.gradio['dry_base'] = gr.Slider(1, 4, value=generate_params['dry_base'], step=0.01, label='dry_base', info='Controls how fast the penalty grows with increasing sequence length.')
                            ui.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=generate_params['repetition_penalty'], step=0.01, label='repetition_penalty')
                            ui.gradio['frequency_penalty'] = gr.Slider(0, 2, value=generate_params['frequency_penalty'], step=0.05, label='frequency_penalty')
                            ui.gradio['presence_penalty'] = gr.Slider(0, 2, value=generate_params['presence_penalty'], step=0.05, label='presence_penalty')
                            ui.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=generate_params['encoder_repetition_penalty'], step=0.01, label='encoder_repetition_penalty')
                            ui.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='no_repeat_ngram_size')
                            ui.gradio['repetition_penalty_range'] = gr.Slider(0, 4096, step=64, value=generate_params['repetition_penalty_range'], label='repetition_penalty_range')

                        with gr.Column():
                            gr.Markdown('## Alternative sampling methods')
                            ui.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='penalty_alpha', info='For Contrastive Search. do_sample must be unchecked.')
                            ui.gradio['guidance_scale'] = gr.Slider(-0.5, 2.5, step=0.05, value=generate_params['guidance_scale'], label='guidance_scale', info='For CFG. 1.5 is a good value.')
                            ui.gradio['mirostat_mode'] = gr.Slider(0, 2, step=1, value=generate_params['mirostat_mode'], label='mirostat_mode', info='mode=1 is for llama.cpp only.')
                            ui.gradio['mirostat_tau'] = gr.Slider(0, 10, step=0.01, value=generate_params['mirostat_tau'], label='mirostat_tau')
                            ui.gradio['mirostat_eta'] = gr.Slider(0, 1, step=0.01, value=generate_params['mirostat_eta'], label='mirostat_eta')

                            gr.Markdown('## Other options')
                            ui.gradio['max_new_tokens'] = gr.Slider(minimum=ui.settings['max_new_tokens_min'], 
                                                                    maximum=ui.settings['max_new_tokens_max'], 
                                                                      value=ui.settings['max_new_tokens'], step=1, label='max_new_tokens', info='⚠️ Setting this too high can cause prompt truncation.')
                            ui.gradio['max_tokens_second'] = gr.Slider(value=ui.settings['max_tokens_second'], minimum=0, maximum=20, step=1, label='#tokens/second', info='To make text readable in real time.')
                            ui.gradio['max_updates_second'] = gr.Slider(value=ui.settings['max_updates_second'], minimum=0, maximum=24, step=1, label='updates/second', info='Set this if you experience lag in the UI during streaming.')
                            ui.gradio['prompt_lookup_num_tokens'] = gr.Slider(value=ui.settings['prompt_lookup_num_tokens'], minimum=0, maximum=10, step=1, label='prompt_lookup_num_tokens', info='Activates Prompt Lookup Decoding.')

                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            ui.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='do_sample')
                            ui.gradio['dynamic_temperature'] = gr.Checkbox(value=generate_params['dynamic_temperature'], label='dynamic_temperature')
                            ui.gradio['temperature_last'] = gr.Checkbox(value=generate_params['temperature_last'], label='temperature_last', info=ui.element_description['params_temperature_last'])
                            ui.gradio['auto_max_new_tokens'] = gr.Checkbox(value=ui.settings['auto_max_new_tokens'], label='auto_max_new_tokens', info='Expand max_new_tokens to the available context length.')
                            ui.gradio['ban_eos_token'] = gr.Checkbox(value=ui.settings['ban_eos_token'], label='Ban the eos_token', info='Forces the model to never end the generation prematurely.')
                            ui.gradio['add_bos_token'] = gr.Checkbox(value=ui.settings['add_bos_token'], label='Add the bos_token', info='Disabling this can make the replies more creative.')
                            ui.gradio['skip_special_tokens'] = gr.Checkbox(value=ui.settings['skip_special_tokens'], label='Skip special tokens', info='Some specific models need this unset.')
                            ui.gradio['static_cache'] = gr.Checkbox(value=ui.settings['static_cache'], label='Static KV cache', info='Use a static cache for improved performance.')
                            ui.gradio['stream'] = gr.Checkbox(value=ui.settings['stream'], label='Activate text streaming')

                        with gr.Column():
                            ui.gradio['truncation_length'] = gr.Number(precision=0, step=256, value=get_truncation_length(), label='Truncate the prompt up to this length', info='The leftmost tokens are removed if the prompt exceeds this length.')
                            ui.gradio['seed'] = gr.Number(value=ui.settings['seed'], label='Seed (-1 for random)')

                            ui.gradio['sampler_priority'     ] = gr.Textbox(value=generate_params['sampler_priority'], lines=12, label='Sampler priority', info='Parameter names separated by new lines or commas.', elem_classes=['add_scrollbar'])
                            ui.gradio['dry_sequence_breakers'] = gr.Textbox(value=generate_params['dry_sequence_breakers'], label='dry_sequence_breakers', info=ui.element_description['params_dry_seq_breakers'])
                            ui.gradio['custom_stopping_strings' ] = gr.Textbox(value=ui.settings["custom_stopping_strings"] or None, lines=2, label='Custom stopping strings', info='Written between "" and separated by commas.', placeholder='"\\n", "\\nYou:"')
                            ui.gradio['custom_token_bans'       ] = gr.Textbox(value=ui.settings['custom_token_bans'] or None, label='Token bans', info=ui.element_description['params_custom_token_bans'])
                            ui.gradio['negative_prompt'         ] = gr.Textbox(value=ui.settings['negative_prompt'], label='Negative prompt', info='For CFG. Only used when guidance_scale is different than 1.', lines=3, elem_classes=['add_scrollbar'])
                            ui.gradio['show_after'              ] = gr.Textbox(value=ui.settings['show_after'] or None, label='Show after', info='Hide the reply before this text.', placeholder="</think>")
                            
                            with gr.Row() as ui.gradio['grammar_file_row']:
                                ui.gradio['grammar_file'] = gr.Dropdown(value='None', choices=all_grammars, label='Load grammar from file (.gbnf)', elem_classes='slim-dropdown')
                                create_refresh_button(
                                    ui.gradio['grammar_file'], 
                                    lambda: None, 
                                    lambda: {'choices': all_grammars}, elem_classes='refresh-button', interactive=is_single_user)
                                
                                ui.gradio['save_grammar'] = gr.Button('💾', elem_classes='refresh-button', interactive=is_single_user)
                                ui.gradio['delete_grammar'] = gr.Button('🗑️ ', elem_classes='refresh-button', interactive=is_single_user)

                            ui.gradio['grammar_string'] = gr.Textbox(value='', label='Grammar', lines=16, elem_classes=['add_scrollbar', 'monospace'])

        from . import ui_chat
        ui_chat.create_chat_settings_ui()


def create_event_handlers():

    preset_params = preset.presets_params()

    ui.gradio['filter_by_loader'].change(
        blacklist_samplers, 
        gradget('filter_by_loader', 'dynamic_temperature'), 
        gradget(list_all_samplers()), 
        show_progress=False)

    ui.gradio['preset_menu'].change(
        gather_interface_values, 
        gradget(ui.input_elements), 
        gradget('interface_state')
    ).then(
        preset.load_preset_for_ui, 
        gradget('preset_menu', 'interface_state'), 
        gradget('interface_state') + gradget(preset_params), 
        show_progress=False
    )

    ui.gradio['random_preset'].click(
        gather_interface_values, 
        gradget(ui.input_elements), 
        gradget('interface_state')
    ).then(
        preset.random_preset, 
        gradget('interface_state'), 
        gradget('interface_state') + gradget(preset_params), 
        show_progress=False
    )

    ui.gradio['grammar_file'].change(
        load_grammar, 
        gradget('grammar_file'), 
        gradget('grammar_string'), 
        show_progress=False)

    ui.gradio['dynamic_temperature'].change(
        lambda x: [gr.update(visible=x)] * 3, 
        gradget('dynamic_temperature'), 
        gradget('dynatemp_low', 'dynatemp_high', 'dynatemp_exponent'), 
        show_progress=False)


def get_truncation_length():
    if 'max_seq_len' in shared.args_provided \
                    or shared.args.max_seq_len != shared.args_default.max_seq_len:
        return shared.args.max_seq_len
    elif 'n_ctx' in shared.args_provided \
                or shared.args.n_ctx != shared.args_default.n_ctx:
        return shared.args.n_ctx
    else:
        return ui.settings['truncation_length']


def load_grammar(name):
    p = Path(f'grammars/{name}')
    if p.exists():
        return open(p, 'r', encoding='utf-8').read()
    else:
        return ''


def blacklist_samplers(loader, dynamic_temperature):
    all_samplers = list_all_samplers()
    output = []

    for sampler in all_samplers:
        if loader == 'All' or sampler in loaders_samplers[loader]:
            if sampler.startswith('dynatemp'):
                output.append(gr.update(visible=dynamic_temperature))
            else:
                output.append(gr.update(visible=True))
        else:
            output.append(gr.update(visible=False))

    return output
