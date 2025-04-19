import json
import gradio as gr

from functools import partial
from pathlib import Path
from PIL import Image

from ..src import chat, shared
from ..src.chat import send_last_reply_to_input, handle_upload_chat_history
from ..src.utils import get_available_chat_styles, get_available_characters, get_available_instructions
from ..src.generation import stop_everything_event

from . import wrapper
from . import shared as ui
from .html import chat_html_wrapper
from .utils import gradget, create_refresh_button, gather_interface_values


def create_ui():
    
    is_single_user = not ui.multi_user
    all_chat_styles = get_available_chat_styles()

    ui.gradio['Chat input'] = gr.State()
    ui.gradio['history'] = gr.JSON(visible=False)

    with gr.Tab('Chat', id='Chat', elem_id='chat-tab'):

        with gr.Row(elem_id='past-chats-row', elem_classes=['pretty_scrollbar']):

            with gr.Column():
                with gr.Row(elem_id='past-chats-buttons'):
                    ui.gradio['branch_chat'] = gr.Button('Branch', elem_classes='refresh-button', interactive=is_single_user)
                    ui.gradio['rename_chat'] = gr.Button('Rename', elem_classes='refresh-button', interactive=is_single_user)
                    ui.gradio['delete_chat'] = gr.Button(  '🗑️' , elem_classes='refresh-button', interactive=is_single_user)
                    ui.gradio['start_chat'] = gr.Button('New chat', elem_classes=['refresh-button', 'focus-on-chat-input'])

                ui.gradio['search_chat'] = gr.Textbox(placeholder='Search chats...', max_lines=1, elem_id='search_chat')

                with gr.Row(elem_id='delete-chat-row', visible=False) as ui.gradio['delete-chat-row']:
                    ui.gradio['delete_chat-cancel'] = gr.Button('Cancel', elem_classes=['refresh-button', 'focus-on-chat-input'])
                    ui.gradio['delete_chat-confirm'] = gr.Button('Confirm', variant='stop', elem_classes=['refresh-button', 'focus-on-chat-input'])

                with gr.Row(elem_id='rename-row', visible=False) as ui.gradio['rename-row']:
                    ui.gradio['rename_to'] = gr.Textbox(label='Rename to:', placeholder='New name', elem_classes=['no-background'])
                    with gr.Row():
                        ui.gradio['rename_to-cancel'] = gr.Button('Cancel', elem_classes=['refresh-button', 'focus-on-chat-input'])
                        ui.gradio['rename_to-confirm'] = gr.Button('Confirm', elem_classes=['refresh-button', 'focus-on-chat-input'], variant='primary')

                with gr.Row():
                    ui.gradio['unique_id'] = gr.Radio(label="", elem_classes=['slim-dropdown', 'pretty_scrollbar'], interactive=is_single_user, elem_id='past-chats')

        with gr.Row():

            with gr.Column(elem_id='chat-col'):
                ui.gradio['html_display'] = gr.HTML(value=chat_html_wrapper({'internal': [], 'visible': []}, '', '', 'chat', 'cai-chat', ''), visible=True)
                ui.gradio['display'] = gr.Textbox(value="", visible=False)  # Hidden buffer
                
                with gr.Row(elem_id="chat-input-row"):

                    with gr.Column(scale=1, elem_id='gr-hover-container'):
                        gr.HTML(value='<div class="hover-element" onclick="void(0)"><span style="width: 100px; display: block" id="hover-element-button">&#9776;</span><div class="hover-menu" id="hover-menu"></div>', elem_id='gr-hover')

                    with gr.Column(scale=10, elem_id='chat-input-container'):
                        ui.gradio['textbox'] = gr.Textbox(label='', placeholder='Send a message', elem_id='chat-input', elem_classes=['add_scrollbar'])
                        ui.gradio['show_controls'] = gr.Checkbox(value=ui.settings['show_controls'], label='Show controls (Ctrl+S)', elem_id='show-controls')
                        ui.gradio['typing-dots'] = gr.HTML(value='<div class="typing"><span></span><span class="dot1"></span><span class="dot2"></span></div>', label='typing', elem_id='typing-container')

                    with gr.Column(scale=1, elem_id='generate-stop-container'):
                        with gr.Row():
                            ui.gradio['Stop'] = gr.Button('Stop', elem_id='stop', visible=False)
                            ui.gradio['Generate'] = gr.Button('Generate', elem_id='Generate', variant='primary')

        # Hover menu buttons
        with gr.Column(elem_id='chat-buttons'):

            with gr.Row():
                ui.gradio['Regenerate'] = gr.Button('Regenerate (Ctrl + Enter)', elem_id='Regenerate')
                ui.gradio[ 'Continue' ] = gr.Button('Continue (Alt + Enter)', elem_id='Continue')
                ui.gradio['Remove last'] = gr.Button('Remove last (Ctrl + Shift + Backspace)', elem_id='Remove-last')

            with gr.Row():
                ui.gradio['Replace last'] = gr.Button('Replace last reply (Ctrl + Shift + L)', elem_id='Replace-last')
                ui.gradio[   'Copy last'] = gr.Button('Copy last reply (Ctrl + Shift + K)', elem_id='Copy-last')
                ui.gradio['Impersonate'] = gr.Button('Impersonate (Ctrl + Shift + M)', elem_id='Impersonate')

            with gr.Row():
                ui.gradio['Send dummy message'] = gr.Button('Send dummy message')
                ui.gradio['Send dummy reply'] = gr.Button('Send dummy reply')

            with gr.Row():
                ui.gradio['send-chat-to-default'] = gr.Button('Send to default')
                ui.gradio['send-chat-to-notebook'] = gr.Button('Send to notebook')

        with gr.Row(elem_id='chat-controls', elem_classes=['pretty_scrollbar']):

            with gr.Column():
                with gr.Row():
                    ui.gradio['start_with'] = gr.Textbox(
                        label='Start reply with', 
                        placeholder='Sure thing!', 
                        value=ui.settings['start_with'], 
                        elem_classes=['add_scrollbar']
                    )

                with gr.Row():
                    ui.gradio['mode'] = gr.Radio(
                        choices=['chat', 'chat-instruct', 'instruct'], 
                        value=ui.settings['mode'] if ui.settings['mode'] in ['chat', 'chat-instruct'] else None, 
                         info=ui.element_description['chat_mode'], 
                        label='Mode', 
                        elem_id='chat-mode'
                    )

                with gr.Row():
                    ui.gradio['chat_style'] = gr.Dropdown(
                        choices=all_chat_styles, 
                          value=ui.settings['chat_style'], 
                        visible=ui.settings['mode'] != 'instruct', 
                          label='Chat style'
                    )

                with gr.Row():
                    ui.gradio['chat-instruct_command'] = gr.Textbox(
                          value=ui.settings['chat-instruct_command'], 
                        visible=ui.settings['mode'] == 'chat-instruct', 
                           info=ui.element_description['chat_instruct_cmd'], 
                          lines=12, label='Command for chat-instruct mode', 
                        elem_classes=['add_scrollbar']
                    )


def create_chat_settings_ui():

    is_single_user = not ui.multi_user
    all_characters = get_available_characters()
    all_instructions = get_available_instructions()

    with gr.Tab('Chat'):

        with gr.Row():

            with gr.Column(scale=8):
    
                with gr.Tab("Character"):
                    with gr.Row():
                        ui.gradio['character_menu'] = gr.Dropdown(label='Character', value=None, choices=all_characters, elem_id='character-menu', elem_classes='slim-dropdown', info='Used in chat and chat-instruct modes.')
                        create_refresh_button(ui.gradio['character_menu'], lambda: None, lambda: {'choices': all_characters}, 'refresh-button', interactive=is_single_user)
                        ui.gradio['save_character'] = gr.Button('💾', elem_classes='refresh-button', elem_id="save-character", interactive=is_single_user)
                        ui.gradio['delete_character'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=is_single_user)

                    ui.gradio['name2'] = gr.Textbox(value='', lines=1, label='Character name')
                    ui.gradio['context'] = gr.Textbox(value='', lines=10, label='Context', elem_classes=['add_scrollbar'])
                    ui.gradio['greeting'] = gr.Textbox(value='', lines=5, label='Greeting', elem_classes=['add_scrollbar'])

                with gr.Tab("User"):
                    ui.gradio['name1'] = gr.Textbox(value=ui.settings['name1'], lines=1, label='Name')
                    ui.gradio['user_bio'] = gr.Textbox(value=ui.settings['user_bio'], lines=10, label='Description', 
                                                            info='Here you can optionally write a description of yourself.', 
                                                     placeholder='{{user}}\'s personality: ...', 
                                                    elem_classes=['add_scrollbar'])

                with gr.Tab('Chat history'):
                    with gr.Row():
                        with gr.Column():
                            ui.gradio['save_chat_history'] = gr.Button(value='Save history')

                        with gr.Column():
                            ui.gradio['load_chat_history'] = gr.File(type='binary', file_types=['.json', '.txt'], label='Upload History JSON')

                with gr.Tab('Upload character'):
                    with gr.Tab('YAML or JSON'):
                        with gr.Row():
                            ui.gradio['upload_json'] = gr.File(type='binary', file_types=['.json', '.yaml'], label='JSON or YAML File', interactive=is_single_user)
                            ui.gradio['upload_img_bot'] = gr.Image(type='pil', label='Profile Picture (optional)', interactive=is_single_user)

                        ui.gradio['Submit character'] = gr.Button(value='Submit', interactive=False)

                    with gr.Tab('TavernAI PNG'):
                        with gr.Row():
                            with gr.Column():
                                ui.gradio['upload_img_tavern'] = gr.Image(type='pil', label='TavernAI PNG File', elem_id='upload_img_tavern', interactive=is_single_user)
                                ui.gradio['tavern_json'] = gr.State()
                            with gr.Column():
                                ui.gradio['tavern_name'] = gr.Textbox(value='', lines=1, label='Name', interactive=False)
                                ui.gradio['tavern_desc'] = gr.Textbox(value='', lines=10, label='Description', interactive=False, elem_classes=['add_scrollbar'])

                        ui.gradio['Submit character (tavern)'] = gr.Button(value='Submit', interactive=False)

            with gr.Column(scale=1):
                self_pic = chat.character_cache_folder / "cache/pfp_me.png"
                ui.gradio['character_picture'] = gr.Image(label='Character picture', type='pil', interactive=is_single_user)
                ui.gradio['personal_picture'] = gr.Image(label='Personal picture', type='pil', interactive=is_single_user, 
                                                        value=Image.open(self_pic) if self_pic.exists() else None)

    with gr.Tab('Instruction template'):

        with gr.Row():
        
            with gr.Column():
                with gr.Row():
                    ui.gradio['instruction_template'] = gr.Dropdown(choices=all_instructions, label='Saved instruction templates', 
                                                                    info="After selecting the template, click on \"Load\" to load and apply it.", value='None', elem_classes='slim-dropdown')
                    create_refresh_button(ui.gradio['instruction_template'], lambda: None, lambda: {'choices': all_instructions}, 'refresh-button', interactive=is_single_user)
                    ui.gradio['load_template'] = gr.Button("Load", elem_classes='refresh-button')
                    ui.gradio['save_template'] = gr.Button('💾', elem_classes='refresh-button', interactive=is_single_user)
                    ui.gradio['delete_template'] = gr.Button('🗑️ ', elem_classes='refresh-button', interactive=is_single_user)

            with gr.Column():
                pass

        with gr.Row():
            
            with gr.Column():
                ui.gradio['custom_system_message'] = gr.Textbox(value=ui.settings['custom_system_message'], 
                                                                            label='Custom system message', lines=2, info=ui.element_description['custom_system_message'], elem_classes=['add_scrollbar'])
                ui.gradio['instruction_template_str'] = gr.Textbox(value='', label='Instruction template', lines=24, info=ui.element_description['instruction_template'], elem_classes=['add_scrollbar', 'monospace'])
                
                with gr.Row():
                    ui.gradio['send_instruction_to_default'] = gr.Button('Send to default', elem_classes=['small-button'])
                    ui.gradio['send_instruction_to_notebook'] = gr.Button('Send to notebook', elem_classes=['small-button'])
                    ui.gradio['send_instruction_to_negative_prompt'] = gr.Button('Send to negative prompt', elem_classes=['small-button'])

            with gr.Column():
                ui.gradio['chat_template_str'] = gr.Textbox(value=ui.settings['chat_template_str'], label='Chat template', lines=22, elem_classes=['add_scrollbar', 'monospace'])


default_inputs = ('Chat input', 'interface_state')
reload_inputs = ('history', 'name1', 'name2', 'mode', 'chat_style', 'character_menu')

def create_event_handlers():

    # Obsolete variables, kept for compatibility with old extensions
    ui.reload_inputs = gradget(reload_inputs)
    ui.input_params = gradget(default_inputs)
    ui.default_inputs = gradget(default_inputs)

    # JavaScript functions
    html_updates_only = "(text) => handleMorphdomUpdate(text)"
    chat_updates_add = '() => document.getElementById("chat").parentNode.parentNode.parentNode.classList.add("_generating")'
    chat_updates_rmv = '() => document.getElementById("chat").parentNode.parentNode.parentNode.classList.remove("_generating")'
    change_char_menu = "(mode) => {mode === 'instruct' ? document.getElementById('character-menu').parentNode.parentNode.style.display = 'none' : document.getElementById('character-menu').parentNode.parentNode.style.display = ''}"

    save_files = \
        f'(hist, char, mode) => {{{ui.js["save_files"]}; saveHistory(hist, char, mode)}}'
    switch_tab_2_chat = f'() => {{{ui.js["switch_tabs"]}; switch_to_chat()}}'
    switch_tab_2_char = f'() => {{{ui.js["switch_tabs"]}; switch_to_character()}}'
    switch_tab_2_auto = f'() => {{{ui.js["switch_tabs"]}; switch_to_default()}}'
    switch_tab_2_nobo = f'() => {{{ui.js["switch_tabs"]}; switch_to_notebook()}}'
    switch_tab_2_para = f'() => {{{ui.js["switch_tabs"]}; switch_to_generation_parameters()}}'
    audio_noti_ring   = f'() => {{{ui.js["audio_noti"]}}}'
    visualz_controls = f'(x) => {{{ui.js["show_controls"]}; toggle_controls(x)}}'
    update_bigpicture = f'() => {{{ui.js["update_big_picture"]}; updateBigPicture()}}'

    # Morph HTML updates instead of updating everything
    ui.gradio['display'].change(None, gradget('display'), None, js=html_updates_only)

    ui.gradio['Generate'].click(
                                fn=gather_interface_values, 
                            inputs=gradget(ui.input_elements), 
                            outputs=gradget('interface_state')
                        ).then( 
                                fn=lambda x: (x, ''), 
                            inputs=gradget('textbox'), 
                            outputs=gradget('Chat input', 'textbox'), 
                            show_progress=False
                        ).then(lambda: None, None, None, js=chat_updates_add
                        ).then(
                                fn=wrapper.generate_chat_reply_wrapper, 
                            inputs=gradget(default_inputs), 
                            outputs=gradget('display', 'history'), 
                            show_progress=False
                        ).then(None, None, None, js=chat_updates_rmv
                        ).then(None, None, None, js=audio_noti_ring)

    ui.gradio['textbox'].submit(
                                fn=gather_interface_values, 
                            inputs=gradget(ui.input_elements), 
                            outputs=gradget('interface_state')
                        ).then(
                                fn=lambda x: (x, ''), 
                            inputs=gradget('textbox'), 
                            outputs=gradget('Chat input', 'textbox'), 
                            show_progress=False
                        ).then(lambda: None, None, None, js=chat_updates_add
                        ).then(
                                fn=wrapper.generate_chat_reply_wrapper, 
                            inputs=gradget(default_inputs), 
                            outputs=gradget('display', 'history'), 
                            show_progress=False
                        ).then(None, None, None, js=chat_updates_rmv
                        ).then(None, None, None, js=audio_noti_ring)

    ui.gradio['Regenerate'].click(
                                fn=gather_interface_values, 
                            inputs=gradget(ui.input_elements), 
                            outputs=gradget('interface_state')
                        ).then(lambda: None, None, None, js=chat_updates_add
                        ).then(
                                fn=partial(wrapper.generate_chat_reply_wrapper, regenerate=True), 
                            inputs=gradget(default_inputs), 
                            outputs=gradget('display', 'history'), 
                            show_progress=False
                        ).then(None, None, None, js=chat_updates_rmv
                        ).then(None, None, None, js=audio_noti_ring)

    ui.gradio['Continue'].click(
                                fn=gather_interface_values, 
                            inputs=gradget(ui.input_elements), 
                            outputs=gradget('interface_state')
                        ).then(lambda: None, None, None, js=chat_updates_add
                        ).then(
                                fn=partial(wrapper.generate_chat_reply_wrapper, _continue=True), 
                            inputs=gradget(default_inputs), 
                            outputs=gradget('display', 'history'), 
                            show_progress=False
                        ).then(None, None, None, js=chat_updates_rmv
                        ).then(None, None, None, js=audio_noti_ring)

    ui.gradio['Impersonate'].click(
                                    fn=gather_interface_values, 
                                inputs=gradget(ui.input_elements), 
                                outputs=gradget('interface_state')
                            ).then(
                                    fn=lambda x: x, 
                                inputs=gradget('textbox'), 
                                outputs=gradget('Chat input'), 
                                show_progress=False
                            ).then(lambda: None, None, None, js=chat_updates_add
                            ).then(
                                    fn=wrapper.impersonate_wrapper, 
                                inputs=gradget(default_inputs), 
                                outputs=gradget('textbox', 'display'), 
                                show_progress=False
                            ).then(None, None, None, js=chat_updates_rmv
                            ).then(None, None, None, js=audio_noti_ring)

    ui.gradio['Replace last'].click(
                                    fn=gather_interface_values, 
                                inputs=gradget(ui.input_elements), 
                                outputs=gradget('interface_state')
                            ).then(
                                    fn=wrapper.handle_replace_last_reply_click, 
                                inputs=gradget('textbox', 'interface_state'), 
                                outputs=gradget('history', 'display', 'textbox'), 
                                show_progress=False)

    ui.gradio['Remove last'].click(
                                    fn=gather_interface_values, 
                                inputs=gradget(ui.input_elements), 
                                outputs=gradget('interface_state')
                            ).then(
                                    fn=wrapper.handle_remove_last_click, 
                                inputs=gradget('interface_state'), 
                                outputs=gradget('history', 'display', 'textbox'), 
                                show_progress=False)

    ui.gradio['Copy last'].click(fn=send_last_reply_to_input, 
                                inputs=gradget('history'), 
                                outputs=gradget('textbox'), 
                                show_progress=False)

    ui.gradio['Send dummy message'].click(
                                            fn=gather_interface_values, 
                                        inputs=gradget(ui.input_elements), 
                                        outputs=gradget('interface_state')
                                    ).then(
                                            fn=wrapper.handle_send_dummy_message_click, 
                                        inputs=gradget('textbox', 'interface_state'), 
                                        outputs=gradget('history', 'display', 'textbox'), 
                                        show_progress=False)

    ui.gradio['Send dummy reply'].click(
                                        fn=gather_interface_values, 
                                    inputs=gradget(ui.input_elements), 
                                    outputs=gradget('interface_state')
                                ).then(
                                        fn=wrapper.handle_send_dummy_reply_click, 
                                    inputs=gradget('textbox', 'interface_state'), 
                                    outputs=gradget('history', 'display', 'textbox'), 
                                    show_progress=False)

    ui.gradio['Stop'].click(stop_everything_event, None, None, queue=False)\
                      .then(fn=chat.redraw_html, 
                            inputs=gradget(reload_inputs), 
                            outputs=gradget('display'), 
                            show_progress=False)

    if not ui.multi_user:
        ui.gradio['unique_id'].select(
                                        fn=gather_interface_values, 
                                    inputs=gradget(ui.input_elements), 
                                    outputs=gradget('interface_state')
                                ).then(
                                        fn=wrapper.handle_unique_id_select, 
                                    inputs=gradget('interface_state'), 
                                    outputs=gradget('history', 'display'), 
                                    show_progress=False)

    ui.gradio['start_chat'].click(
                                    fn=gather_interface_values, 
                                inputs=gradget(ui.input_elements), 
                                outputs=gradget('interface_state')
                            ).then(
                                    fn=wrapper.handle_start_new_chat_click, 
                                inputs=gradget('interface_state'), 
                                outputs=gradget('history', 'display', 'unique_id'), 
                                show_progress=False)

    ui.gradio['delete_chat'       ].click(lambda: gr.update(visible=True), None, gradget('delete-chat-row'))
    ui.gradio['delete_chat-cancel'].click(lambda: gr.update(visible=False), None, gradget('delete-chat-row'))
    ui.gradio['delete_chat-confirm'].click(
                                            fn=gather_interface_values, 
                                        inputs=gradget(ui.input_elements), 
                                        outputs=gradget('interface_state')
                                    ).then(
                                            fn=wrapper.handle_delete_chat_confirm_click, 
                                        inputs=gradget('interface_state'), 
                                        outputs=gradget('history', 'display', 'unique_id', 'delete-chat-row'), 
                                        show_progress=False)

    ui.gradio['branch_chat'].click(
                                    fn=gather_interface_values, 
                                inputs=gradget(ui.input_elements), 
                                outputs=gradget('interface_state')
                            ).then(
                                    fn=wrapper.handle_branch_chat_click, 
                                inputs=gradget('interface_state'), 
                                outputs=gradget('history', 'display', 'unique_id'), 
                                show_progress=False)

    ui.gradio['rename_chat'].click(wrapper.handle_rename_chat_click, None, gradget('rename_to', 'rename-row'), show_progress=False)
    ui.gradio['rename_to-cancel'].click(lambda: gr.update(visible=False), None, gradget('rename-row'), show_progress=False)
    ui.gradio['rename_to-confirm'].click(
                                            fn=gather_interface_values, 
                                        inputs=gradget(ui.input_elements), 
                                        outputs=gradget('interface_state')
                                    ).then(
                                            fn=wrapper.handle_rename_chat_confirm, 
                                        inputs=gradget('rename_to', 'interface_state'), 
                                        outputs=gradget('unique_id', 'rename-row'))

    ui.gradio['rename_to'].submit(
                                    fn=gather_interface_values, 
                                inputs=gradget(ui.input_elements), 
                                outputs=gradget('interface_state')
                            ).then(
                                    fn=wrapper.handle_rename_chat_confirm, 
                                inputs=gradget('rename_to', 'interface_state'), 
                                outputs=gradget('unique_id', 'rename-row'), 
                                show_progress=False)

    ui.gradio['search_chat'].change(
                                    fn=gather_interface_values, 
                                inputs=gradget(ui.input_elements),
                                outputs=gradget('interface_state')
                            ).then(
                                    fn=wrapper.handle_search_chat_change, 
                                inputs=gradget('interface_state'), 
                                outputs=gradget('unique_id'), 
                                show_progress=False)

    ui.gradio['load_chat_history'].upload(
                                        fn=gather_interface_values, 
                                    inputs=gradget(ui.input_elements), 
                                    outputs=gradget('interface_state')
                                ).then(
                                        fn=handle_upload_chat_history, 
                                    inputs=gradget('load_chat_history', 'interface_state'), 
                                    outputs=gradget('history', 'display', 'unique_id'), 
                                    show_progress=False
                                ).then(None, None, None, js=switch_tab_2_chat)

    ui.gradio['character_menu'].change(
                                        fn=gather_interface_values, 
                                    inputs=gradget(ui.input_elements), 
                                    outputs=gradget('interface_state')
                                ).then(
                                        fn=wrapper.handle_character_menu_change, 
                                    inputs=gradget('interface_state'), 
                                    outputs=gradget('history', 'display', 'name1', 'name2', 'character_picture', 'greeting', 'context', 'unique_id'), 
                                    show_progress=False
                                ).then(None, None, None, js=update_bigpicture)

    ui.gradio['mode'].change(
                                fn=gather_interface_values, 
                            inputs=gradget(ui.input_elements), 
                            outputs=gradget('interface_state')
                        ).then(
                                fn=wrapper.handle_mode_change, 
                            inputs=gradget('interface_state'), 
                            outputs=gradget('history', 'display', 'chat_style', 'chat-instruct_command', 'unique_id'), 
                            show_progress=False
                        ).then(None, gradget('mode'), None, js=change_char_menu)

    ui.gradio['chat_style'].change(fn=chat.redraw_html, 
                                inputs=gradget(reload_inputs), 
                                outputs=gradget('display'), 
                                show_progress=False)

    # Save/delete a character
    ui.gradio['save_character'].click(
                                    fn=wrapper.handle_save_character_click, 
                                inputs=gradget('name2'), 
                                outputs=gradget('save_character_filename', 'character_saver'), 
                                show_progress=False)

    ui.gradio['delete_character'].click(
                                    fn=lambda: gr.update(visible=True), 
                                inputs=None, 
                                outputs=gradget('character_deleter'), 
                                show_progress=False)

    ui.gradio['load_template'].click(
                                    fn=wrapper.handle_load_template_click, 
                                inputs=gradget('instruction_template'), 
                                outputs=gradget('instruction_template_str', 'instruction_template'), 
                                show_progress=False)

    ui.gradio['save_template'].click(
                                    fn=gather_interface_values, 
                                inputs=gradget(ui.input_elements), 
                                outputs=gradget('interface_state')
                            ).then(
                                    fn=wrapper.handle_save_template_click, 
                                inputs=gradget('instruction_template_str'), 
                                outputs=gradget('save_filename', 'save_root', 'save_contents', 'file_saver'), 
                                show_progress=False)

    ui.gradio['delete_template'].click(
                                    fn=wrapper.handle_delete_template_click, 
                                inputs=gradget('instruction_template'), 
                                outputs=gradget('delete_filename', 'delete_root', 'file_deleter'), 
                                show_progress=False)

    ui.gradio['save_chat_history'].click(
                                        fn=lambda x: json.dumps(x, indent=4), 
                                    inputs=gradget('history'), 
                                    outputs=gradget('temporary_text')
                                ).then(None, gradget('temporary_text', 'character_menu', 'mode'), 
                                       None, js=save_files)

    ui.gradio['Submit character'].click(
                                        fn=chat.upload_character, 
                                    inputs=gradget('upload_json', 'upload_img_bot'), 
                                    outputs=gradget('character_menu'), 
                                    show_progress=False
                                ).then(None, None, None, js=switch_tab_2_char)

    ui.gradio['Submit character (tavern)'].click(
                                        chat.upload_tavern_character, 
                                        inputs=gradget('upload_img_tavern', 'tavern_json'), 
                                        outputs=gradget('character_menu'), 
                                        show_progress=False
                                    ).then(None, None, None, js=switch_tab_2_char)

    ui.gradio['upload_json'].upload(lambda: gr.update(interactive=True), None, gradget('Submit character'))
    ui.gradio['upload_json'].clear(lambda: gr.update(interactive=False), None, gradget('Submit character'))
    
    ui.gradio['upload_img_tavern'].upload(fn=chat.check_tavern_character, 
                                        inputs=gradget('upload_img_tavern'), 
                                        outputs=gradget('tavern_name', 'tavern_desc', 'tavern_json', 'Submit character (tavern)'), 
                                        show_progress=False)

    ui.gradio['upload_img_tavern'].clear(fn=lambda: (None, None, None, gr.update(interactive=False)), 
                                        inputs=None, 
                                        outputs=gradget('tavern_name', 'tavern_desc', 'tavern_json', 'Submit character (tavern)'), 
                                        show_progress=False)

    ui.gradio['personal_picture'].change(
                                        fn=gather_interface_values, 
                                    inputs=gradget(ui.input_elements), 
                                    outputs=gradget('interface_state')
                                ).then(
                                        fn=wrapper.handle_profile_change, 
                                    inputs=gradget('personal_picture', 'interface_state'), 
                                    outputs=gradget('display'), 
                                    show_progress=False)

    ui.gradio['send_instruction_to_default'].click(
                                                    fn=gather_interface_values, 
                                                inputs=gradget(ui.input_elements), 
                                                outputs=gradget('interface_state')
                                            ).then(
                                                    fn=wrapper.handle_send_instruction_click, 
                                                inputs=gradget('interface_state'), 
                                                outputs=gradget('textbox-default'), 
                                                show_progress=False
                                            ).then(None, None, None, js=switch_tab_2_auto)

    ui.gradio['send_instruction_to_notebook'].click(
                                                    fn=gather_interface_values, 
                                                inputs=gradget(ui.input_elements), 
                                                outputs=gradget('interface_state')
                                            ).then(
                                                    fn=wrapper.handle_send_instruction_click, 
                                                inputs=gradget('interface_state'), 
                                                outputs=gradget('textbox-notebook'), 
                                                show_progress=False
                                            ).then(None, None, None, js=switch_tab_2_nobo)

    ui.gradio['send_instruction_to_negative_prompt'].click(
                                                            fn=gather_interface_values, 
                                                        inputs=gradget(ui.input_elements), 
                                                        outputs=gradget('interface_state')
                                                    ).then(
                                                            fn=wrapper.handle_send_instruction_click, 
                                                        inputs=gradget('interface_state'), 
                                                        outputs=gradget('negative_prompt'), 
                                                        show_progress=False
                                                    ).then(None, None, None, js=switch_tab_2_para)

    ui.gradio['send-chat-to-default'].click(
                                            fn=gather_interface_values, 
                                        inputs=gradget(ui.input_elements), 
                                        outputs=gradget('interface_state')
                                    ).then(
                                            fn=wrapper.handle_send_chat_click, 
                                        inputs=gradget('interface_state'), 
                                        outputs=gradget('textbox-default'), 
                                        show_progress=False
                                    ).then(None, None, None, js=switch_tab_2_auto)

    ui.gradio['send-chat-to-notebook'].click(
                                            fn=gather_interface_values, 
                                        inputs=gradget(ui.input_elements), 
                                        outputs=gradget('interface_state')
                                    ).then(
                                            fn=wrapper.handle_send_chat_click, 
                                        inputs=gradget('interface_state'), 
                                        outputs=gradget('textbox-notebook'), 
                                        show_progress=False
                                    ).then(None, None, None, js=switch_tab_2_nobo)

    ui.gradio['show_controls'].change(None, gradget('show_controls'), None, js=visualz_controls)

