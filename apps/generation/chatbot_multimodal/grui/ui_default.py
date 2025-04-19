import gradio as gr

from . import shared as ui
from .utils import gradget, create_refresh_button, gather_interface_values, current_time

from ..src import logits
from ..src.utils import get_available_prompts
from ..src.prompts import count_tokens, load_prompt, prompt_dir
from ..src.generation import generate_reply_wrapper, get_token_ids, stop_everything_event


inputs = ('textbox-default', 'interface_state')
outputs = ('output_textbox', 'html-default')


def create_ui():

    is_single_user = not ui.multi_user
    all_prompts = get_available_prompts()

    with gr.Tab('Default', elem_id='default-tab'):
    
        with gr.Row():

            with gr.Column():

                with gr.Row():
                    ui.gradio['textbox-default'] = gr.Textbox(value='', lines=27, label='Input', elem_classes=['textbox_default', 'add_scrollbar'])
                    ui.gradio['token-counter-default'] = gr.HTML(value="<span>0</span>", elem_id="default-token-counter")

                with gr.Row():
                    ui.gradio['Continue-default'] = gr.Button('Continue')
                    ui.gradio['Stop-default'] = gr.Button('Stop', elem_id='stop', visible=False)
                    ui.gradio['Generate-default'] = gr.Button('Generate', variant='primary')

                with gr.Row():
                    ui.gradio['prompt_menu-default'] = gr.Dropdown(choices=all_prompts, value='None', label='Prompt', elem_classes='slim-dropdown')
                    create_refresh_button(
                        ui.gradio['prompt_menu-default'], 
                        lambda: None, lambda: {'choices': all_prompts}, 
                        'refresh-button', 
                        interactive=is_single_user
                    )

                    ui.gradio['save_prompt-default'] = gr.Button('💾', elem_classes='refresh-button', interactive=is_single_user)
                    ui.gradio['delete_prompt-default'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=is_single_user)

            with gr.Column():
                with gr.Tab('Raw'):
                    ui.gradio['output_textbox'] = gr.Textbox(lines=27, label='Output', elem_id='textbox-default', 
                                                                                    elem_classes=['textbox_default_output', 'add_scrollbar'])

                with gr.Tab('Markdown'):
                    ui.gradio['markdown_render-default'] = gr.Button('Render')
                    ui.gradio['markdown-default'] = gr.Markdown()

                with gr.Tab('HTML'):
                    ui.gradio['html-default'] = gr.HTML()

                with gr.Tab('Logits'):
                    with gr.Row():
                        with gr.Column(scale=10):
                            ui.gradio['get_logits-default'] = gr.Button('Get next token probabilities')
                        with gr.Column(scale=1):
                            ui.gradio['use_samplers-default'] = gr.Checkbox(label='Use samplers', value=True, elem_classes=['no-background'])

                    with gr.Row():
                        ui.gradio['logits-default'         ] = gr.Textbox(lines=23, label=         'Output', elem_classes=['textbox_logits', 'add_scrollbar'])
                        ui.gradio['logits-default-previous'] = gr.Textbox(lines=23, label='Previous output', elem_classes=['textbox_logits', 'add_scrollbar'])

                with gr.Tab('Tokens'):
                    ui.gradio['get_tokens-default'] = gr.Button('Get token IDs for the input')
                    ui.gradio['tokens-default'] = gr.Textbox(lines=23, label='Tokens', elem_classes=['textbox_logits', 'add_scrollbar', 'monospace'])


def create_event_handlers():

    # JavaScript functions
    audio_noti_ring = f'() => {{{ui.js["audio_noti"]}}}'

    # Gradio functions
    ui.gradio['Generate-default'].click(
                                    gather_interface_values, 
                                    gradget(ui.input_elements), 
                                    gradget('interface_state')
                                ).then(
                                    lambda: [gr.update(visible=True), gr.update(visible=False)], 
                                    None, 
                                    gradget('Stop-default', 'Generate-default')
                                ).then(
                                    generate_reply_wrapper, 
                                    gradget(inputs), 
                                    gradget(outputs), 
                                    show_progress=False
                                ).then(
                                    lambda state, left, right: state.update({'textbox-default': left, 'output_textbox': right}), 
                                    gradget('interface_state', 'textbox-default', 'output_textbox'), 
                                    None
                                ).then(
                                    lambda: [gr.update(visible=False), gr.update(visible=True)], 
                                    None, 
                                    gradget('Stop-default', 'Generate-default')
                                ).then(None, None, None, js=audio_noti_ring)

    ui.gradio['textbox-default'].submit(
                                    gather_interface_values, 
                                    gradget(ui.input_elements), 
                                    gradget('interface_state')
                                ).then(
                                    lambda: [gr.update(visible=True), gr.update(visible=False)], 
                                    None, 
                                    gradget('Stop-default', 'Generate-default')
                                ).then(
                                    generate_reply_wrapper, 
                                    gradget(inputs), 
                                    gradget(outputs), 
                                    show_progress=False
                                ).then(
                                    lambda state, left, right: state.update({'textbox-default': left, 'output_textbox': right}), 
                                    gradget('interface_state', 'textbox-default', 'output_textbox'), 
                                    None
                                ).then(
                                    lambda: [gr.update(visible=False), gr.update(visible=True)], 
                                    None, 
                                    gradget('Stop-default', 'Generate-default')
                                ).then(None, None, None, js=audio_noti_ring)

    ui.gradio['Continue-default'].click(
                                    gather_interface_values, 
                                    gradget(ui.input_elements), 
                                    gradget('interface_state')
                                ).then(
                                    lambda: [gr.update(visible=True), gr.update(visible=False)], 
                                    None, 
                                    gradget('Stop-default', 'Generate-default')
                                ).then(
                                    generate_reply_wrapper, 
                                    [ui.gradio['output_textbox']] + gradget(inputs)[1:], 
                                    gradget(outputs), 
                                    show_progress=False
                                ).then(
                                    lambda state, left, right: state.update({'textbox-default': left, 'output_textbox': right}), 
                                    gradget('interface_state', 'textbox-default', 'output_textbox'), 
                                    None
                                ).then(
                                    lambda: [gr.update(visible=False), gr.update(visible=True)], 
                                    None, 
                                    gradget('Stop-default', 'Generate-default')
                                ).then(None, None, None, js=audio_noti_ring)

    ui.gradio['Stop-default'].click(stop_everything_event, None, None, queue=False)

    ui.gradio['markdown_render-default'].click(
        lambda x: x, 
        gradget('output_textbox'), 
        gradget('markdown-default'), 
        queue=False
    )

    ui.gradio['prompt_menu-default'].change(
        load_prompt, 
        gradget('prompt_menu-default'), 
        gradget('textbox-default'), 
        show_progress=False
    )

    ui.gradio['save_prompt-default'].click(
        handle_save_prompt, 
        gradget('textbox-default'), 
        gradget('save_contents', 'save_filename', 'save_root', 'file_saver'), 
        show_progress=False
    )

    ui.gradio['delete_prompt-default'].click(
        handle_delete_prompt, 
        gradget('prompt_menu-default'), 
        gradget('delete_filename', 'delete_root', 'file_deleter'), 
        show_progress=False
    )

    ui.gradio['textbox-default'].change(
        lambda x: f"<span>{count_tokens(x)}</span>", 
        gradget('textbox-default'), 
        gradget('token-counter-default'), 
        show_progress=False
    )

    ui.gradio['get_logits-default'].click(
        gather_interface_values, 
        gradget(ui.input_elements), 
        gradget('interface_state')
    ).then(
        logits.get_next_logits, 
        gradget('textbox-default', 'interface_state', 'use_samplers-default', 'logits-default'), 
        gradget('logits-default', 'logits-default-previous'), 
        show_progress=False
    )

    ui.gradio['get_tokens-default'].click(
        get_token_ids, 
        gradget('textbox-default'), 
        gradget('tokens-default'), 
        show_progress=False
    )


def handle_save_prompt(text):
    return [text, f"{current_time()}.txt", str(prompt_dir), gr.update(visible=True)]


def handle_delete_prompt(prompt):
    return [f"{prompt}.txt", str(prompt_dir), gr.update(visible=True)]
