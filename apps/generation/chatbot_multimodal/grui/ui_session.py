import gradio as gr
from pathlib import Path

from . import shared as ui
from .utils import gradget, list_model_elements, get_available_extensions, gather_interface_values

from ..src import shared


def create_ui():

    is_single_user = not ui.multi_user
    all_extensions = get_available_extensions()
    all_bool_args = get_boolean_arguments()
    active_bool_args = get_boolean_arguments(active=True)

    with gr.Tab("Session", elem_id="session-tab"):

        with gr.Row():
    
            with gr.Column():
                ui.gradio['reset_interface'] = gr.Button("Apply and restart", interactive=is_single_user)

                with gr.Row():
                    ui.gradio['toggle_dark_mode'] = gr.Button('Toggle 💡')
                    ui.gradio['save_settings'] = gr.Button('Save UI', interactive=is_single_user)

                with gr.Row():
                    with gr.Column():
                        ui.gradio[
                            'extensions_menu'] = gr.CheckboxGroup(choices=all_extensions, value=ui.extensions, label="Available extensions", elem_classes='checkboxgroup-table', info=ui.element_description['extension_menu'])

                    with gr.Column():
                        ui.gradio['bool_menu'] = gr.CheckboxGroup(choices=all_bool_args, value=active_bool_args, label="Boolean CMD flags", elem_classes='checkboxgroup-table')

            with gr.Column():
                extension_name = gr.Textbox(lines=1, label='Install or update', interactive=is_single_user, info=ui.element_description['extension_install'])
                extension_status = gr.Markdown()

        ui.gradio['theme_state'] = gr.Textbox(visible=False, value='dark' if ui.settings['dark_theme'] else 'light')


def create_event_handlers():

    # JavaScript
    reload_js = '() => {document.body.innerHTML=\'<h1 style="font-family:monospace;padding-top:20%;margin:0;height:100vh;color:lightgray;text-align:center;background:var(--body-background-fill)">Reloading...</h1>\'; setTimeout(function(){location.reload()},2500); return []}'
    dark_theme = f'() => {{{ui.js["dark_theme"]}; toggleDarkMode()}}'

    # Reset interface event
    ui.gradio['reset_interface'].click(
        set_interface_arguments, 
        gradget('extensions_menu', 'bool_menu'), 
        None
    ).then(None, None, None, js=reload_js)

    ui.gradio['toggle_dark_mode'].click(
        lambda x: 'dark' if x == 'light' else 'light', 
        gradget('theme_state'), 
        gradget('theme_state')
    ).then(
        None, None, None, js=dark_theme)

    ui.gradio['save_settings'].click(
        gather_interface_values, 
        gradget(ui.input_elements), 
        gradget('interface_state')
    ).then(
        handle_save_settings, 
        gradget('interface_state', 'preset_menu', 'extensions_menu', 'show_controls', 'theme_state'), 
        gradget('save_contents', 'save_filename', 'save_root', 'file_saver'), 
        show_progress=False
    )


def handle_save_settings(state, preset, extensions, show_controls, theme):
    return [
        utils.save_settings(state, preset, extensions, show_controls, theme), 
        "settings.yaml", 
        str(Path(__file__).parents[1] / "src/config"), 
        gr.update(visible=True)
    ]


def set_interface_arguments(extensions, bool_active):

    bool_list = get_boolean_arguments()
    for k in bool_list:
        setattr(shared.args, k, False)
    for k in bool_active:
        setattr(shared.args, k, True)

    ui.extensions = extensions
    ui.need_restart = True


def get_boolean_arguments(active=False):

    exclude = shared.args_deprecated
    cmd_list = vars(shared.args)
    bool_list = sorted([k for k in cmd_list 
                           if type(cmd_list[k]) is bool and k not in (exclude + list_model_elements())])
    bool_active = [k for k in bool_list if vars(shared.args)[k]]
    if active:
        return bool_active
    else:
        return bool_list

