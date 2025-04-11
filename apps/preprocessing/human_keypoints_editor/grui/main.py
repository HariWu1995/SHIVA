import gradio as gr
from threading import Thread

from .ui_2d import create_ui as create_ui_2d, serve_local_editor as serve_2d_editor
from .ui_3d import create_ui as create_ui_3d, serve_local_editor as serve_3d_editor


def create_ui(min_width: int = 25):

    gui_config = dict(css=None, analytics_enabled=False)

    with gr.Blocks(**gui_config) as gui:

        with gr.Tab(label='2D Editor') as gui_auto:
            create_ui_2d()

        with gr.Tab(label='3D Editor') as gui_point:
            create_ui_3d()

    return gui, None


if __name__ == "__main__":

    server_2ditor = Thread(target=serve_2d_editor, daemon=True)
    server_3ditor = Thread(target=serve_3d_editor, daemon=True)
    server_2ditor.start()
    server_3ditor.start()

    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000)

