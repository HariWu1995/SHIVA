import os
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

import matplotlib
matplotlib.use('Agg')  # This fixes LaTeX rendering on some systems

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='The value passed into gr.Dropdown()')
warnings.filterwarnings('ignore', category=UserWarning, message='Using the update method is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='Field "model_name" has conflict')
warnings.filterwarnings('ignore', category=UserWarning, message='Field "model_names" has conflict')

from .logging import logger
from .requests_block import OpenMonkeyPatch, RequestBlocker
with RequestBlocker():
    from . import gradio_hijack
    import gradio as gr

import sys
import signal

def signal_handler(sig, frame):
    logger.info("Received Ctrl+C. Shutting down Text generation web UI gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


import time
from pathlib import Path
from functools import partial
from threading import Lock, Thread

import json
import yaml

from . import (
    shared as ui, 
    ui_default, ui_chat, ui_file, ui_model, ui_notebook, ui_session, ui_params,
    extensions,
)
from .utils import (
    get_available_extensions, 
    get_fallback_settings, 
    gradget, 
    update_model_parameters,
    list_interface_input_elements,
    apply_interface_values,
)

from ..src import shared
from ..src.loaders import get_model_metadata, load_model, unload_model_if_idle


# JavaScript

def interface_launch_js(js_combined):
    return f"""(x) => {{
        if ({str(ui.settings['dark_theme']).lower()}) {{
            document.getElementsByTagName('body')[0].classList.add('dark');
        }}
        else {{
            document.getElementsByTagName('body')[0].classList.remove('dark');
        }}
        {js_combined}
        {ui.js["show_controls"]}
        toggle_controls(x);
    }}"""


# Define UI settings & layout

def create_ui(min_width: int = 25):

    # Import the extensions and execute their setup() functions
    if ui.extensions is not None and \
    len(ui.extensions) > 0:
        extensions.load_extensions()

    # Force some events to be triggered on page load
    ui.persistent_interface_state.update({
        'loader': shared.loader or 'Transformers',
        'mode': ui.settings['mode'] if ui.settings['mode'] == 'instruct' else gr.update(),
        'character_menu': shared.args.character or ui.settings['character'],
        'instruction_template_str': ui.settings['instruction_template_str'],
        'prompt_menu-default': ui.settings['prompt-default'],
        'prompt_menu-notebook': ui.settings['prompt-notebook'],
        'filter_by_loader': shared.loader or 'All',
    })

    # css/js strings
    css = ui.css
    js = ui.js['main']
    css += extensions.apply_extensions('css')
    js += extensions.apply_extensions('js')

    # Interface state elements
    ui.input_elements = list_interface_input_elements()

    title = "Dialogue with SHIVA"
    interface_kwargs = dict(css=css, analytics_enabled=False, title=title, theme=ui.theme)
    
    with gr.Blocks(**interface_kwargs) as ui.gradio['interface']:

        # Interface state
        ui.gradio['interface_state'] = gr.State({k: None for k in ui.input_elements})

        # Audio notification
        if os.path.isfile(ui.audio_noti_filepath):
            ui.gradio['audio_notification'] = gr.Audio(
                value=ui.audio_noti_filepath, 
                elem_id="audio_notification", interactive=False, visible=False
            )

        # Floating menus for saving / deleting files
        ui_file.create_ui()

        # Temporary clipboard for saving files
        ui.gradio['temporary_text'] = gr.Textbox(visible=False)

        # Load tabs
        ui_chat.create_ui()
        ui_default.create_ui()
        ui_notebook.create_ui()
        ui_model.create_ui()
        ui_session.create_ui()
        ui_params.create_ui(ui.settings['preset'])
        # ui_train.create_ui()

        # Generation events
        ui_chat.create_event_handlers()
        ui_default.create_event_handlers()
        ui_notebook.create_event_handlers()
        ui_file.create_event_handlers()
        ui_model.create_event_handlers()
        ui_session.create_event_handlers()
        ui_params.create_event_handlers()

        # Interface launch events
        ui.gradio['interface'].load(None, gradget('show_controls'), None, js=interface_launch_js(js))
        ui.gradio['interface'].load(
            partial(apply_interface_values, {}, use_persistent=True), None, gradget(ui.input_elements), show_progress=False)

        extensions.create_extensions_tabs()
        extensions.create_extensions_block()
    
    return ui.gradio['interface']


def run():
    gui = create_ui()
    gui.queue()

    with OpenMonkeyPatch():
        gui.launch(
            share=False,
            server_name='0.0.0.0',
            server_port=8000,
            inbrowser=ui.auto_launch,
        )


if __name__ == "__main__":

    shared.generation_lock = Lock()

    # if shared.args.idle_timeout > 0:
    #     timer_thread = Thread(target=unload_model_if_idle)
    #     timer_thread.daemon = True
    #     timer_thread.start()

    # Launch the web UI
    run()
    while True:
        time.sleep(0.5)
        if shared.need_restart:
            shared.need_restart = False
            time.sleep(0.5)
            ui.gradio['interface'].close()
            time.sleep(0.5)
            run()

