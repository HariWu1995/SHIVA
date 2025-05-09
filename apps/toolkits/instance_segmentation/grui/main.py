import gradio as gr

from .ui_auto import create_ui as create_ui_auto
from .ui_point import create_ui as create_ui_point
from .ui_prompt import create_ui as create_ui_prompt


def create_ui(min_width: int = 25):

    gui_config = dict(css=None, analytics_enabled=False)

    with gr.Blocks(**gui_config) as gui:

        with gr.Tab(label='Auto') as gui_auto:
            create_ui_auto()

        with gr.Tab(label='Point') as gui_point:
            create_ui_point()

        with gr.Tab(label='Prompt') as gui_prompt:
            create_ui_prompt()

    return gui, None


if __name__ == "__main__":
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000)

