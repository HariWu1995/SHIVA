import traceback
import gradio as gr

from . import shared as ui
from .utils import gradget, gather_interface_values

from ..src import chat
from ..src.preset import generate_preset_yaml

from ..src import utils
from ..src.utils.default import PRESET_DIR, GRAMMAR_DIR


def create_ui():

    is_single_user = not ui.multi_user

    # Text file saver
    with gr.Group(visible=False, elem_classes='file-saver') as ui.gradio['file_saver']:
        ui.gradio['save_filename'] = gr.Textbox(lines=1, label='File name')
        ui.gradio['save_root'    ] = gr.Textbox(lines=1, label='File folder', info='For reference. Unchangeable.', interactive=False)
        ui.gradio['save_contents'] = gr.Textbox(lines=10, label='File contents')

        with gr.Row():
            ui.gradio['save_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            ui.gradio['save_confirm'] = gr.Button('Save', elem_classes="small-button", variant='primary', interactive=is_single_user)

    # Text file deleter
    with gr.Group(visible=False, elem_classes='file-saver') as ui.gradio['file_deleter']:
        ui.gradio['delete_filename'] = gr.Textbox(lines=1, label='File name')
        ui.gradio['delete_root'    ] = gr.Textbox(lines=1, label='File folder', info='For reference. Unchangeable.', interactive=False)
        
        with gr.Row():
            ui.gradio['delete_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            ui.gradio['delete_confirm'] = gr.Button('Delete', elem_classes="small-button", variant='stop', interactive=is_single_user)

    # Character saver/deleter
    with gr.Group(visible=False, elem_classes='file-saver') as ui.gradio['character_saver']:
        ui.gradio['save_character_filename'] = gr.Textbox(lines=1, label='File name', info='The character will be saved to your characters/ folder with this base filename.')
        with gr.Row():
            ui.gradio['save_character_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            ui.gradio['save_character_confirm'] = gr.Button('Save', elem_classes="small-button", variant='primary', interactive=is_single_user)

    with gr.Group(visible=False, elem_classes='file-saver') as ui.gradio['character_deleter']:
        gr.Markdown('Confirm the character deletion?')
        with gr.Row():
            ui.gradio['delete_character_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            ui.gradio['delete_character_confirm'] = gr.Button('Delete', elem_classes="small-button", variant='stop', interactive=is_single_user)

    # Preset saver
    with gr.Group(visible=False, elem_classes='file-saver') as ui.gradio['preset_saver']:
        ui.gradio['save_preset_filename'] = gr.Textbox(lines=1, label='File name', info='The preset will be saved to your presets/ folder with this base filename.')
        ui.gradio['save_preset_contents'] = gr.Textbox(lines=10, label='File contents')
        with gr.Row():
            ui.gradio['save_preset_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            ui.gradio['save_preset_confirm'] = gr.Button('Save', elem_classes="small-button", variant='primary', interactive=is_single_user)


def create_event_handlers():

    ui.gradio['save_preset'].click(
        gather_interface_values, 
        gradget(ui.input_elements), 
        gradget('interface_state')
    ).then(
        handle_save_preset_click, 
        gradget('interface_state'), 
        gradget('save_preset_contents', 'save_preset_filename', 'preset_saver'), 
        show_progress=False
    )

    ui.gradio['delete_preset'].click(
        handle_delete_preset_click, 
        gradget('preset_menu'), 
        gradget('delete_filename', 'delete_root', 'file_deleter'), 
        show_progress=False
    )

    ui.gradio['save_grammar'].click(
        handle_save_grammar_click, 
        gradget('grammar_string'), 
        gradget('save_contents', 'save_filename', 'save_root', 'file_saver'), 
        show_progress=False
    )

    ui.gradio['delete_grammar'].click(
        handle_delete_grammar_click, 
        gradget('grammar_file'), 
        gradget('delete_filename', 'delete_root', 'file_deleter'),
        show_progress=False
    )

    ui.gradio['save_preset_confirm'].click(
        handle_save_preset_confirm_click, 
        gradget('save_preset_filename', 'save_preset_contents'), 
        gradget('preset_menu', 'preset_saver'), 
        show_progress=False
    )

    ui.gradio['save_confirm'].click(
        handle_save_confirm_click, 
        gradget('save_root', 'save_filename', 'save_contents'), 
        gradget('file_saver'), 
        show_progress=False
    )

    ui.gradio['delete_confirm'].click(
        handle_delete_confirm_click, 
        gradget('delete_root', 'delete_filename'), 
        gradget('file_deleter'), 
        show_progress=False
    )

    ui.gradio['save_character_confirm'].click(
        handle_save_character_confirm_click, 
        gradget('name2', 'greeting', 'context', 'character_picture', 'save_character_filename'), 
        gradget('character_menu', 'character_saver'), 
        show_progress=False
    )

    ui.gradio['delete_character_confirm'].click(
        handle_delete_character_confirm_click, 
        gradget('character_menu'), 
        gradget('character_menu', 'character_deleter'), 
        show_progress=False
    )

    ui.gradio['save_preset_cancel'      ].click(lambda: gr.update(visible=False), None, gradget('preset_saver'), show_progress=False)
    ui.gradio['save_cancel'             ].click(lambda: gr.update(visible=False), None, gradget('file_saver'))
    ui.gradio['delete_cancel'           ].click(lambda: gr.update(visible=False), None, gradget('file_deleter'))
    ui.gradio['save_character_cancel'   ].click(lambda: gr.update(visible=False), None, gradget('character_saver'), show_progress=False)
    ui.gradio['delete_character_cancel' ].click(lambda: gr.update(visible=False), None, gradget('character_deleter'), show_progress=False)


def handle_save_preset_confirm_click(filename, contents):
    try:
        utils.save_file(PRESET_DIR / f"{filename}.yaml", contents)
        available_presets = utils.get_available_presets()
        output = gr.update(choices=available_presets, value=filename)
    except Exception:
        output = gr.update()
        traceback.print_exc()
    return [output, gr.update(visible=False)]


def handle_save_confirm_click(root, filename, contents):
    try:
        utils.save_file(root + filename, contents)
    except Exception:
        traceback.print_exc()
    return gr.update(visible=False)


def handle_delete_confirm_click(root, filename):
    try:
        utils.delete_file(root + filename)
    except Exception:
        traceback.print_exc()
    return gr.update(visible=False)


def handle_save_character_confirm_click(name2, greeting, context, character_picture, filename):
    try:
        chat.save_character(name2, greeting, context, character_picture, filename)
        available_characters = utils.get_available_characters()
        output = gr.update(choices=available_characters, value=filename)
    except Exception:
        output = gr.update()
        traceback.print_exc()
    return [output, gr.update(visible=False)]


def handle_delete_character_confirm_click(character):
    try:
        index = str(utils.get_available_characters().index(character))
        chat.delete_character(character)
        output = chat.update_character_menu_after_deletion(index)
    except Exception:
        output = gr.update()
        traceback.print_exc()
    return [output, gr.update(visible=False)]


def handle_save_preset_click(state):
    contents = generate_preset_yaml(state)
    return [contents, "My Preset", gr.update(visible=True)]


def handle_delete_preset_click(preset):
    return [f"{preset}.yaml", str(PRESET_DIR), gr.update(visible=True)]


def handle_save_grammar_click(grammar_string):
    return [grammar_string, "My Fancy Grammar.gbnf", str(GRAMMAR_DIR), gr.update(visible=True)]


def handle_delete_grammar_click(grammar_file):
    return [grammar_file, str(GRAMMAR_DIR), gr.update(visible=True)]

