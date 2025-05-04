import gradio as gr

from . import shared as ui
from .ui_ws_gen  import create_ui as create_ui_gen
from .ui_ws_var  import create_ui as create_ui_var
from .ui_ws_fill import create_ui as create_ui_fill


def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:
        create_ui_gen()
        create_ui_var()
        create_ui_fill()        

    return gui


if __name__ == "__main__":
    gui= create_ui()
    gui.launch(server_name='localhost', server_port=8000)

