import gradio as gr

from . import shared as ui
from .utils import gradget, gather_interface_values, create_refresh_button
from .ui_default import handle_delete_prompt, handle_save_prompt

from ..src.utils import get_available_prompts
from ..src.logits import get_next_logits
from ..src.prompts import count_tokens, load_prompt
from ..src.generation import get_token_ids, generate_reply_wrapper, stop_everything_event


inputs = ('textbox-notebook', 'interface_state')
outputs = ('textbox-notebook', 'html-notebook')


def create_ui():

    is_single_user = not ui.multi_user

    with gr.Tab('Notebook', elem_id='notebook-tab'):

        ui.gradio['last_input-notebook'] = gr.State('')

        with gr.Row():

            with gr.Column(scale=4):
            
                with gr.Tab('Raw'):
                    with gr.Row():
                        ui.gradio['textbox-notebook'] = gr.Textbox(value='', lines=27, elem_id='textbox-notebook', elem_classes=['textbox', 'add_scrollbar'])
                        ui.gradio['token-counter-notebook'] = gr.HTML(value="<span>0</span>", elem_id="notebook-token-counter")

                with gr.Tab('Markdown'):
                    ui.gradio['markdown_render-notebook'] = gr.Button('Render')
                    ui.gradio['markdown-notebook'] = gr.Markdown()

                with gr.Tab('HTML'):
                    ui.gradio['html-notebook'] = gr.HTML()

                with gr.Tab('Logits'):
                    with gr.Row():
                        with gr.Column(scale=10):
                            ui.gradio['get_logits-notebook'] = gr.Button('Get next token probabilities')
                        with gr.Column(scale=1):
                            ui.gradio['use_samplers-notebook'] = gr.Checkbox(label='Use samplers', value=True, elem_classes=['no-background'])

                    with gr.Row():
                        ui.gradio['logits-notebook'] = gr.Textbox(lines=23, label='Output', elem_classes=['textbox_logits_notebook', 'add_scrollbar'])
                        ui.gradio['logits-notebook-previous'] = gr.Textbox(lines=23, label='Previous output', elem_classes=['textbox_logits_notebook', 'add_scrollbar'])

                with gr.Tab('Tokens'):
                    ui.gradio['get_tokens-notebook'] = gr.Button('Get token IDs for the input')
                    ui.gradio['tokens-notebook'] = gr.Textbox(lines=23, label='Tokens', elem_classes=['textbox_logits_notebook', 'add_scrollbar', 'monospace'])

                with gr.Row():
                    ui.gradio['Undo'] = gr.Button('Undo', elem_classes='small-button')
                    ui.gradio['Regenerate-notebook'] = gr.Button('Regenerate', elem_classes='small-button')
                    ui.gradio['Stop-notebook'] = gr.Button('Stop', visible=False, elem_classes='small-button', elem_id='stop')
                    ui.gradio['Generate-notebook'] = gr.Button('Generate', variant='primary', elem_classes='small-button')

            with gr.Column(scale=1):
                gr.HTML('<div style="padding-bottom: 13px"></div>')
                with gr.Row():
                    ui.gradio['prompt_menu-notebook'] = gr.Dropdown(choices=get_available_prompts(), value='None', label='Prompt', elem_classes='slim-dropdown')
                    create_refresh_button(
                        ui.gradio['prompt_menu-notebook'], 
                        lambda: None, 
                        lambda: {'choices': get_available_prompts()}, elem_classes=['refresh-button', 'refresh-button-small'], interactive=is_single_user)

                    ui.gradio['save_prompt-notebook'] = gr.Button('💾', elem_classes=['refresh-button', 'refresh-button-small'], interactive=is_single_user)
                    ui.gradio['delete_prompt-notebook'] = gr.Button('🗑️', elem_classes=['refresh-button', 'refresh-button-small'], interactive=is_single_user)


def create_event_handlers():

    audio_noti_ring = f'() => {{{ui.js["audio_noti"]}}}'

    ui.gradio['Generate-notebook'].click(
                                        fn=lambda x: x, 
                                    inputs=gradget('textbox-notebook'), 
                                    outputs=gradget('last_input-notebook')
                                ).then(
                                        fn=gather_interface_values, 
                                    inputs=gradget(ui.input_elements), 
                                    outputs=gradget('interface_state')
                                ).then(
                                        fn=lambda: [gr.update(visible=True), gr.update(visible=False)], 
                                    inputs=None, 
                                    outputs=gradget('Stop-notebook', 'Generate-notebook')
                                ).then(
                                        fn=generate_reply_wrapper, 
                                    inputs=gradget(inputs), 
                                    outputs=gradget(outputs), 
                                    show_progress=False
                                ).then(
                                        fn=lambda state, text: state.update({'textbox-notebook': text}), 
                                    inputs=gradget('interface_state', 'textbox-notebook'), 
                                    outputs=None
                                ).then(
                                        fn=lambda: [gr.update(visible=False), gr.update(visible=True)], 
                                    inputs=None, 
                                    outputs=gradget('Stop-notebook', 'Generate-notebook')
                                ).then(None, None, None, js=audio_noti_ring)

    ui.gradio['textbox-notebook'].submit(
                                        fn=lambda x: x, 
                                    inputs=gradget('textbox-notebook'), 
                                    outputs=gradget('last_input-notebook')
                                ).then(
                                        fn=gather_interface_values, 
                                    inputs=gradget(ui.input_elements), 
                                    outputs=gradget('interface_state')
                                ).then(
                                        fn=lambda: [gr.update(visible=True), gr.update(visible=False)],
                                    inputs=None, 
                                    outputs=gradget('Stop-notebook', 'Generate-notebook')
                                ).then(
                                        fn=generate_reply_wrapper, 
                                    inputs=gradget(inputs), 
                                    outputs=gradget(outputs), 
                                    show_progress=False
                                ).then(
                                        fn=lambda state, text: state.update({'textbox-notebook': text}), 
                                    inputs=gradget('interface_state', 'textbox-notebook'), 
                                    outputs=None
                                ).then(
                                        fn=lambda: [gr.update(visible=False), gr.update(visible=True)], 
                                    inputs=None, 
                                    outputs=gradget('Stop-notebook', 'Generate-notebook')
                                ).then(None, None, None, js=audio_noti_ring)

    ui.gradio['Regenerate-notebook'].click(
                                            fn=lambda x: x, 
                                        inputs=gradget('last_input-notebook'), 
                                        outputs=gradget('textbox-notebook'), 
                                        show_progress=False
                                    ).then(
                                            fn=gather_interface_values, 
                                        inputs=gradget(ui.input_elements), 
                                        outputs=gradget('interface_state')
                                    ).then(
                                            fn=lambda: [gr.update(visible=True), gr.update(visible=False)], 
                                        inputs=None, 
                                        outputs=gradget('Stop-notebook', 'Generate-notebook')
                                    ).then(
                                            fn=generate_reply_wrapper, 
                                        inputs=gradget(inputs), 
                                        outputs=gradget(outputs), 
                                        show_progress=False
                                    ).then(
                                            fn=lambda state, text: state.update({'textbox-notebook': text}), 
                                        inputs=gradget('interface_state', 'textbox-notebook'), 
                                        outputs=None
                                    ).then(
                                            fn=lambda: [gr.update(visible=False), gr.update(visible=True)], 
                                        inputs=None, 
                                        outputs=gradget('Stop-notebook', 'Generate-notebook')
                                    ).then(None, None, None, js=audio_noti_ring)

    ui.gradio['Undo'].click(
                            fn=lambda x: x, 
                        inputs=gradget('last_input-notebook'), 
                        outputs=gradget('textbox-notebook'), 
                        show_progress=False
                    ).then(
                            fn=lambda state, text: state.update({'textbox-notebook': text}), 
                        inputs=gradget('interface_state', 'textbox-notebook'), 
                        outputs=None
                    )

    ui.gradio['markdown_render-notebook'].click(
        lambda x: x, 
        gradget('textbox-notebook'), 
        gradget('markdown-notebook'), 
        queue=False
    )

    ui.gradio['Stop-notebook'].click(stop_everything_event, None, None, queue=False)

    ui.gradio['prompt_menu-notebook'].change(
        load_prompt, 
        gradget('prompt_menu-notebook'), 
        gradget('textbox-notebook'), 
        show_progress=False
    )

    ui.gradio['save_prompt-notebook'].click(
        handle_save_prompt, 
        gradget('textbox-notebook'), 
        gradget('save_contents', 'save_filename', 'save_root', 'file_saver'), 
        show_progress=False
    )

    ui.gradio['delete_prompt-notebook'].click(
        handle_delete_prompt, 
        gradget('prompt_menu-notebook'), 
        gradget('delete_filename', 'delete_root', 'file_deleter'), 
        show_progress=False
    )

    ui.gradio['textbox-notebook'].input(
        lambda x: f"<span>{count_tokens(x)}</span>", 
        gradget('textbox-notebook'), 
        gradget('token-counter-notebook'), 
        show_progress=False
    )

    ui.gradio['get_logits-notebook'].click(
                                            fn=gather_interface_values, 
                                        inputs=gradget(ui.input_elements), 
                                        outputs=gradget('interface_state')
                                    ).then(
                                            fn=get_next_logits, 
                                        inputs=gradget('textbox-notebook', 'interface_state', 'use_samplers-notebook', 'logits-notebook'), 
                                        outputs=gradget('logits-notebook', 'logits-notebook-previous'), 
                                        show_progress=False)

    ui.gradio['get_tokens-notebook'].click(
        get_token_ids, 
        gradget('textbox-notebook'), 
        gradget('tokens-notebook'), 
        show_progress=False
    )

