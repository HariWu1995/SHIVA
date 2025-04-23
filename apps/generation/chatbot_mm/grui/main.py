"""
Emojis: 🗣️🗪🗫🗯💭
"""
import os
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

import matplotlib
matplotlib.use('Agg')  # This fixes LaTeX rendering on some systems

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from .utils import OpenMonkeyPatch, RequestBlocker
with RequestBlocker():
    from . import hijack
    import gradio as gr

import sys
import signal

def signal_handler(sig, frame):
    print("\n\nReceived Ctrl+C. \nShutting down WebUI gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


from ..src import shared
from threading import Lock, Thread

shared.generation_lock = Lock()
if shared.model_args.idle_timeout > 0:
    timer_thread = Thread(target=unload_model_if_idle)
    timer_thread.daemon = True
    timer_thread.start()


#############################################
#           UI settings & layout            #
#############################################

from . import shared as ui
from .ui_model import create_ui as create_ui_model
from .ui_dialog import create_ui as create_ui_dialog


def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:
        gr.Markdown("## 🗫 Multi-Modal Dialogue")
        
        with gr.Tab(label='Model Settings') as gui_model:
            create_ui_model()

        with gr.Tab(label='Dialogue') as gui_dialog:
            create_ui_dialog()

    return gui


if __name__ == "__main__":
    gui = create_ui()
    gui.launch(
          server_name = ui.host, 
          server_port = ui.port, 
                share = ui.share,
            inbrowser = ui.auto_launch,
        allowed_paths = ui.allowed_paths,
    )

