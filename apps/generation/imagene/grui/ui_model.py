"""
Emojis: 🗣️🗪🗫🗯💭
"""
from glob import glob
import gradio as gr

from . import shared as ui
from .utils import gradget, symbols

from ..src import shared
from ..src.path import MLM_LOCAL_MODELS, LORA_LOCAL_MODELS, MULTIMODAL_ALLOWED
from ..src.models import MODELLIB_LOADERS, load_model, reload_model, unload_model, unload_model_if_idle
from ..src.utils import get_all_device_memory


def _load_model(model_name, model_type='auto'):
    shared.tokenizer, \
    shared.generator = load_model(model_name, model_type)
    return shared.model_name, shared.model_type, True


def _unload_model():
    unload_model(keep_model_info=True)
    return shared.model_name, shared.model_type, False


def _reload_model():
    reload_model()
    return shared.model_name, shared.model_type, True


#############################################
#           UI settings & layout            #
#############################################

def create_ui(min_width: int = 25):

    all_models = list(MLM_LOCAL_MODELS.keys())
    all_loaders = MODELLIB_LOADERS + ['auto']
    # all_loras = LORA_LOCAL_MODELS
    all_loras = ["test-1","test-2","test-3"]

    # Generation parameters
    all_grammars = get_available_grammars()
    all_presets = get_available_presets()
    all_generams = list(default_preset().keys())
    default_generams = default_preset()

    # Layout
    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        with gr.Accordion(label="Model Loader", open=True):
            with gr.Row(variant="panel"):
                with gr.Column(min_width=50):
                    ui.gradio["model_name"] = gr.Dropdown(label="Model", choices=all_models, interactive=True, value=None)
                    with gr.Row(variant="panel"):
                        ui.gradio["model_status"] = gr.Checkbox(label="Model is loaded", interactive=False)
                    with gr.Row():
                        with gr.Column(min_width=3):
                            ui.gradio["model_load_refresh"] = gr.Button(value=symbols["refresh"], variant="secondary")
                        with gr.Column(min_width=7):
                            ui.gradio["model_load_trigger"] = gr.Button(value=symbols["trigger"], variant="primary")
                        with gr.Column(min_width=3):
                            ui.gradio["model_load_release"] = gr.Button(value=symbols["release"], variant="secondary")
                with gr.Column(min_width=50):
                    ui.gradio["model_type"] = gr.Dropdown(label="Loader", choices=all_loaders, interactive=True, value='auto')
                with gr.Column(min_width=50):
                    ui.gradio["lora_names"] = gr.Dropdown(label="LoRAs", choices=all_loras, interactive=True, multiselect=True)

        with gr.Accordion(label="Generation Parameters", open=False):

            with gr.Tab("Generation"):
                with gr.Row():

                    with gr.Column(min_width=25):
                        ui.gradio['truncation_length'] =   gr.Number(value=ui.settings['truncation_length'], precision=0, step=256, label='truncation_length', info='Number of leftmost tokens are removed if prompt exceeds this length.', interactive=True)
                        ui.gradio['negative_prompt'  ] =  gr.Textbox(value=ui.settings['negative_prompt'], label='Negative prompt', info='For CFG. Only used when guidance_scale is different than 1.', lines=3, interactive=True)
                        ui.gradio['seed'             ] =   gr.Number(value=ui.settings['seed'], label='Seed (-1 for random)', interactive=True)
                        ui.gradio['stream'           ] = gr.Checkbox(value=ui.settings['stream'], label='Activate text streaming', interactive=True)
                        ui.gradio['static_cache'     ] = gr.Checkbox(value=ui.settings['static_cache'], label='Static KV cache', info='Use a static cache for improved performance.', interactive=True)

                    with gr.Column(min_width=25):
                        ui.gradio['max_new_tokens'] = gr.Slider(minimum=ui.settings['max_new_tokens_min'], 
                                                                maximum=ui.settings['max_new_tokens_max'], 
                                                                  value=ui.settings['max_new_tokens'], step=1, label='max_new_tokens', info='⚠️ Setting this too high can cause prompt truncation.', interactive=True)
                        ui.gradio['max_tokens_second'] = gr.Slider(value=ui.settings['max_tokens_second'], minimum=0, maximum=20, step=1, label='#tokens/second', info='To make text readable in real time.', interactive=True)
                        ui.gradio['max_updates_second'] = gr.Slider(value=ui.settings['max_updates_second'], minimum=0, maximum=24, step=1, label='updates/second', info='Set this if you experience lag in the UI during streaming.', interactive=True)
                        ui.gradio['prompt_lookup_num_tokens'] = gr.Slider(value=ui.settings['prompt_lookup_num_tokens'], minimum=0, maximum=10, step=1, label='prompt_lookup_num_tokens', info='Activates Prompt Lookup Decoding.', interactive=True)

                    with gr.Column(min_width=25):
                        ui.gradio['auto_max_new_tokens'] = gr.Checkbox(value=ui.settings['auto_max_new_tokens'], label='auto_max_new_tokens', info='Expand max_new_tokens to the available context length.', interactive=True)
                        ui.gradio['skip_special_tokens'] = gr.Checkbox(value=ui.settings['skip_special_tokens'], label='Skip special tokens', info='Some specific models need this unset.', interactive=True)
                        ui.gradio['custom_token_bans'] = gr.Textbox(value=ui.settings['custom_token_bans'] or None, label='Token bans', info=ui.element_description['params_custom_token_bans'], interactive=True)
                        ui.gradio['custom_stop_strs' ] = gr.Textbox(value=ui.settings["custom_stop_strs"] or None, label='Custom stopping strings', info='Written between "" and separated by commas.', placeholder='"\\n", "\\nYou:"', lines=2, interactive=True)
                        ui.gradio['ban_eos_token'] = gr.Checkbox(value=ui.settings['ban_eos_token'], label='Ban the eos_token', info='Forces the model to never end the generation prematurely.', interactive=True)
                        ui.gradio['add_bos_token'] = gr.Checkbox(value=ui.settings['add_bos_token'], label='Add the bos_token', info='Disabling this can make the replies more creative.', interactive=True)
                        
                    with gr.Column(min_width=25):
                        ui.gradio['sampler_priority'] = gr.Textbox(value=default_generams['sampler_priority'], lines=12, label='Sampler priority', info='Parameter names separated by new lines or commas.', interactive=True)
                        ui.gradio['show_after'] = gr.Textbox(value=ui.settings['show_after'] or None, label='Show after', info='Hide the reply before this text.', placeholder="</think>", interactive=True)
                        ui.gradio['grammar_file'] = gr.Dropdown(value='None', choices=['None']+all_grammars, label='Grammar file (.gbnf)', interactive=True)
                        ui.gradio['grammar_string'] =  gr.Textbox(value='', label='Grammar', lines=10, interactive=False)

            with gr.Tab("Sampling & Diversity"):
                with gr.Row():

                    with gr.Column(min_width=25):
                        gr.Markdown("#### Basic Sampling")
                        ui.gradio['temperature'] =   gr.Slider(value=default_generams['temperature'], label='temperature', min=0.01, max=5, step=0.01, interactive=True, info="Randomness in token selection. Lower = more deterministic; higher = more creative.")
                        ui.gradio['top_k'] = gr.Slider(0, 200, value=default_generams['top_k'], label='top_k', step=1, interactive=True)
                        ui.gradio['top_p'] = gr.Slider(0., 1., value=default_generams['top_p'], label='top_p', step=.01, interactive=True, info="Select the smallest set of tokens with cumulative probability exceeds top-P. Balanced coherence and diversity.")
                        ui.gradio['min_p'] = gr.Slider(0., 1., value=default_generams['min_p'], label='min_p', step=.01, interactive=True, info="Select only tokens with a min probability are considered. Cut off weak candidates.")
                        ui.gradio['typical_p'] =   gr.Slider(value=default_generams['typical_p'], label='typical_p', min=0, max=1, step=0.01, interactive=True, info="Sample tokens close to the expected distribution.")
                        ui.gradio['do_sample'] = gr.Checkbox(value=default_generams['do_sample'], label='do_sample', container=True, interactive=True)

                    with gr.Column(min_width=25):
                        gr.Markdown("#### Dynamic / Adaptive Sampling")
                        ui.gradio['dynatemp_low'       ] =   gr.Slider(value=default_generams['dynatemp_low'], label='dynatemp_low', min=0.01, max=5, step=0.01, interactive=True)
                        ui.gradio['dynatemp_high'      ] =   gr.Slider(value=default_generams['dynatemp_high'], label='dynatemp_high', min=0.01, max=5, step=0.01, interactive=True)
                        ui.gradio['dynatemp_exponent'  ] =   gr.Slider(value=default_generams['dynatemp_exponent'], label='dynatemp_exponent', min=0.01, max=5, step=0.01, interactive=True)
                        ui.gradio['dynamic_temperature'] = gr.Checkbox(value=default_generams['dynamic_temperature'], label='dynamic_temperature', interactive=True, container=True)
                        ui.gradio['temperature_last'   ] = gr.Checkbox(value=default_generams['temperature_last'], label='temperature_last', interactive=True, container=True)
                        ui.gradio['tfs'                ] =   gr.Slider(value=default_generams['tfs'], step=0.01, label='tfs', min=0.0, max=1.0, info='Tail-free sampling: tail of low-probability tokens in distribution', interactive=True)

                    with gr.Column(min_width=25):
                        gr.Markdown("#### Advanced Filtering")
                        ui.gradio['top_a'      ] = gr.Slider(0,  1, value=default_generams['top_a'], step=0.01, label='top_a', interactive=True)
                        ui.gradio['top_n_sigma'] = gr.Slider(0,  5, value=default_generams['top_n_sigma'], step=0.1, label='top_n_sigma', interactive=True)
                        ui.gradio['eps_cutoff' ] = gr.Slider(0,  9, value=default_generams['eps_cutoff'], step=0.1, label='eps_cutoff', interactive=True)
                        ui.gradio['eta_cutoff' ] = gr.Slider(0, 20, value=default_generams['eta_cutoff'], step=0.1, label='eta_cutoff', interactive=True)

                    with gr.Column(min_width=25):
                        gr.Markdown("#### Misc.")
                        ui.gradio['xtc_threshold']   = gr.Slider(value=default_generams['xtc_threshold'], min=0, max=0.5, step=0.01, label='xtc_threshold', info=ui.element_description['params_xtc_threshold'], interactive=True)
                        ui.gradio['xtc_probability'] = gr.Slider(value=default_generams['xtc_probability'],min=0, max=1,  step=0.01, label='xtc_probability', info=ui.element_description['params_xtc_probability'], interactive=True)
                        ui.gradio['guidance_scale']  = gr.Slider(value=default_generams['guidance_scale'], min=-0.5, max=2.5, step=0.1, label='guidance_scale', info='For CFG. 1.5 is a good value.', interactive=True)
                        ui.gradio['smoothing_curve' ] = gr.Slider(1, 10, value=default_generams['smoothing_curve'], step=0.01, label='smoothing_curve', info='Adjusts the dropoff curve of Quadratic Sampling.', interactive=True)
                        ui.gradio['smoothing_factor'] = gr.Slider(0, 10, value=default_generams['smoothing_factor'], step=0.01, label='smoothing_factor', info='Activates Quadratic Sampling.', interactive=True)

            with gr.Tab("Repetition & Perplexity"):
                with gr.Row():

                    with gr.Column(min_width=25, scale=2):
                        gr.Markdown("#### Dry for de-repetition")
                        ui.gradio['dry_sequence_breakers'] = gr.Textbox(value=default_generams['dry_sequence_breakers'], label='dry_sequence_breakers', info=ui.element_description['params_dry_seq_breakers'], interactive=True)
                        ui.gradio['dry_multiplier'] = gr.Slider(0,  5, value=default_generams['dry_multiplier'], step=.1, label='dry_multiplier', info='Set to greater than 0 to enable DRY. Recommended: 0.8.', interactive=True)
                        ui.gradio['dry_length'    ] = gr.Slider(1, 20, value=default_generams['dry_length'], step=.1, label='dry_length', info='Longest sequence that can be repeated without being penalized.', interactive=True)
                        ui.gradio['dry_base'      ] = gr.Slider(1,  4, value=default_generams['dry_base'], step=.1, label='dry_base', info='Controls how fast the penalty grows with increasing sequence length.', interactive=True)

                    with gr.Column(min_width=25, scale=2):
                        gr.Markdown("#### Repetition & Penalty")
                        ui.gradio[   'penalty_alpha'  ] = gr.Slider(0,    5, value=default_generams[   'penalty_alpha'  ], step=0.1, label='penalty_alpha', info='For Contrastive Search. do_sample must be unchecked.', interactive=True)
                        ui.gradio['repetition_penalty'] = gr.Slider(1., 1.5, value=default_generams['repetition_penalty'], step=0.1, label='repetition_penalty', interactive=True)
                        ui.gradio[ 'frequency_penalty'] = gr.Slider( 0,   2, value=default_generams[ 'frequency_penalty'], step=0.1, label='frequency_penalty', interactive=True)
                        ui.gradio[  'presence_penalty'] = gr.Slider( 0,   2, value=default_generams[  'presence_penalty'], step=0.1, label='presence_penalty', interactive=True)
                        ui.gradio['encoder_repetition_penalty'] = gr.Slider(.8, 1.5, value=default_generams['encoder_repetition_penalty'], step=.1, label='encoder_repetition_penalty', interactive=True)
                        ui.gradio[ 'repetition_penalty_range' ] = gr.Slider(0, 4096, value=default_generams['repetition_penalty_range'], step=64, label='repetition_penalty_range', interactive=True)
                        ui.gradio[   'no_repeat_ngram_size'   ] = gr.Slider(0,   20, value=default_generams['no_repeat_ngram_size'], step=1, label='no_repeat_ngram_size', interactive=True)

                    with gr.Column(min_width=25, scale=1):
                        gr.Markdown("#### Mirostat for perplexity")
                        ui.gradio['mirostat_mode'] = gr.Dropdown(choices=[0,1,2], value=default_generams['mirostat_mode'], label='mirostat_mode', info='mode=1 is for llama.cpp only.', interactive=True)
                        ui.gradio['mirostat_tau' ] = gr.Slider(0, 10, step=.1, value=default_generams['mirostat_tau'], label='mirostat_tau', interactive=True)
                        ui.gradio['mirostat_eta' ] = gr.Slider(0,  1, step=.1, value=default_generams['mirostat_eta'], label='mirostat_eta', interactive=True)

        # Recap for later gathering
        ui.generation_params = [
            'temperature','temperature_last','tfs',
            'top_k','top_p','min_p','typical_p','do_sample',
            'top_a','top_n_sigma','eps_cutoff','eta_cutoff',
            'dynatemp_low','dynatemp_high','dynatemp_exponent','dynamic_temperature',
            'truncation_length','negative_prompt','seed','stream','static_cache',
            'max_new_tokens','max_tokens_second','max_updates_second','prompt_lookup_num_tokens',
            'auto_max_new_tokens','skip_special_tokens','ban_eos_token','add_bos_token',
            'dry_length','dry_base','dry_multiplier','dry_sequence_breakers',
            'xtc_threshold','xtc_probability','guidance_scale','smoothing_curve','smoothing_factor',
            'mirostat_mode','mirostat_tau','mirostat_eta',
            'penalty_alpha','encoder_repetition_penalty','repetition_penalty_range','no_repeat_ngram_size',
            'repetition_penalty','frequency_penalty','presence_penalty',
        ]

        ## Event handlers
        m_inputs = gradget("model_name", "model_type")
        m_output = gradget("model_name", "model_type", "model_status")

        ui.gradio["model_load_trigger"].click(fn=_load_model, inputs=m_inputs, outputs=m_output)
        ui.gradio["model_load_release"].click(fn=_unload_model, inputs=None, outputs=m_output)
        ui.gradio["model_load_refresh"].click(fn=_reload_model, inputs=None, outputs=m_output)

        # Schedule
        timer = gr.Timer(value=1.0)
        timer.tick(fn=lambda: shared.generator is not None, outputs=gradget("model_status"))

    return gui


if __name__ == "__main__":
    gui = create_ui()
    gui.launch(
          server_name = ui.host, 
          server_port = ui.port, 
                share = ui.share,
            inbrowser = ui.auto_launch,
    )

