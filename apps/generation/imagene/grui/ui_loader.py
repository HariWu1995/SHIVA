import gradio as gr

from . import shared as ui
from .ui_model import create_ui as create_ui_model
from .ui_loras import create_ui as create_ui_loras


def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:
        create_ui_model()
        create_ui_loras()

    return gui


if __name__ == "__main__":
    gui= create_ui()
    gui.launch(server_name='localhost', server_port=8000)

