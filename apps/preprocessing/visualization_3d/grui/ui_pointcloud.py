from pathlib import Path
from threading import Thread
from multiprocessing import Process

import time
import gradio as gr

from ..src.viz_pointcloud import main as viz_pointcloud


HOST = "localhost"
PORT = 5000
URL = f"http://localhost:{PORT}"

iframe = """
<iframe id="viser3d_iframe" src="{url}" width="100%" height="768"></iframe>
<script>
    window.onload = function() {{
        setTimeout(function() {{
            var iframe = document.getElementById("viser3d_iframe");
            iframe.src = "{url}";
        }}, 500); // wait 500ms to ensure Viser is ready
    }};
</script>
""".format(url=URL)

warning = """
<div style="padding: 10px; border: 1px solid #f0ad4e; border-radius: 8px; background-color: #fcf8e3;">
    ⚠️ <strong>Note:</strong> 
        If the app hasn’t loaded properly, please try refreshing the page. 🔄<br>
        Sometimes a quick refresh is all it takes! 😊
</div>
"""

server_process = None

def serve_local_view(data_path, note):
    if isinstance(data_path, (list, tuple)):
        data_path = data_path[0]
    data_path = Path(data_path)

    global server_process
    if  server_process is not None \
    and server_process.is_alive():
        server_process.terminate()
        server_process.join()
        time.sleep(1)  # Ensure the process has terminated

    # Start a new server process
    server_config = dict(data_path=data_path, host=HOST, port=PORT)
    server_process = Process(target=viz_pointcloud, kwargs=server_config)
    server_process.start()
    time.sleep(5)

    print(f"\nServing {data_path} at {URL}")
    return iframe, gr.Markdown(warning)


# Define UI settings & layout

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    sample_dir = Path(__file__).resolve().parents[4] / "_samples" / "3d"
    sample_path = str(sample_dir / "record3d_dance")
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 🧊 Point-cloud Visualization")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    target = gr.FileExplorer(value=sample_path, root_dir=sample_dir, 
                                            label="Record3D Folder", max_height=250)
                    # target = gr.File(value=sample_path, label="Record3D Folder", file_count="directory")
                with gr.Row():
                    button = gr.Button(value="Run Server", variant="primary")
                with gr.Row():
                    note = gr.Markdown()
            with gr.Column(scale=10):
                display = gr.HTML(label="Viser3D iFrame", value=iframe, show_label=True)

        button.click(fn=serve_local_view, inputs=[target, note], outputs=[display, note])

    return gui, None


if __name__ == "__main__":
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000)

