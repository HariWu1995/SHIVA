import gradio as gr
from threading import Thread

from .ui_smpl import create_ui as create_ui_smpl
from .ui_urdf import create_ui as create_ui_urdf
from .ui_colmap import create_ui as create_ui_colmap
from .ui_pointcloud import create_ui as create_ui_pointcloud
from .ui_gausssplats import create_ui as create_ui_gausssplats

from .utils import find_available_ports


all_ports = find_available_ports(count=10)
local_host = "localhost"


def create_ui(min_width: int = 25):

    gui_config = dict(css=None, analytics_enabled=False)

    with gr.Blocks(**gui_config) as gui:

        with gr.Tab(label='Gauss-Splats') as gui_pausssplats:
            create_ui_gausssplats(host=local_host, port=all_ports.pop(0))

        with gr.Tab(label='Point-Cloud') as gui_pointcloud:
            create_ui_pointcloud(host=local_host, port=all_ports.pop(0))

        with gr.Tab(label='COLMAP') as gui_colmap:
            create_ui_colmap(host=local_host, port=all_ports.pop(0))

        with gr.Tab(label='SMPL') as gui_smpl:
            create_ui_smpl(host=local_host, port=all_ports.pop(0))

        with gr.Tab(label='URDF') as gui_urdf:
            create_ui_urdf(host=local_host, port=all_ports.pop(0))

    return gui, None


if __name__ == "__main__":
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000)
