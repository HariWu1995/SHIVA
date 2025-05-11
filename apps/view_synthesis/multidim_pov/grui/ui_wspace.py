import gradio as gr

from . import shared as ui
from .ui_ws_i2p import create_ui as create_ui_img2pano
from .ui_ws_t2p import create_ui as create_ui_txt2pano
from .ui_ws_p2p import create_ui as create_ui_pano2pano
from .ui_panview import create_ui as create_ui_panview


def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        with gr.Tab(label="Img2Pano"):
            ui.gradio["tab_img2pano"] = create_ui_img2pano()

        with gr.Tab(label="Txt2Pano"):
            ui.gradio["tab_txt2pano"] = create_ui_txt2pano()

        with gr.Tab(label="Upscale"):
            ui.gradio["tab_pano2pano"] = create_ui_pano2pano()

        with gr.Tab(label="365-Viewer"):
            ui.gradio["pano365_viewer"] = create_ui_panview()        

    return gui


if __name__ == "__main__":
    gui= create_ui()
    gui.launch(server_name='localhost', server_port=8000)

