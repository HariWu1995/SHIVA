from pathlib import Path
import gradio as gr

import os
from threading import Thread
from multiprocessing import Process

from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer


root_path = Path(__file__).resolve().parents[0]

HOST = "localhost"
PORT = 3333

local_path = root_path / "gausssplats" / "index.html"
local_url = "http://{host}:{port}/index.html"

remote_url_default = "https://antimatter15.com/splat/"
remote_url_truck = "https://antimatter15.com/splat/?url=truck.splat"
remote_url_stump = "https://antimatter15.com/splat/?url=stump.splat#[-0.86,-0.23,0.45,0,0.27,0.54,0.8,0,-0.43,0.81,-0.4,0,0.92,-2.02,4.1,1]"
remote_url_garden = "https://antimatter15.com/splat/?url=garden.splat"
remote_url_treehill = "https://antimatter15.com/splat/?url=treehill.splat"
remote_url_bicycle = "https://antimatter15.com/splat/?url=bicycle.splat"
remote_url_plush = "https://antimatter15.com/splat/?url=plush.splat#[0.95,0.19,-0.23,0,-0.16,0.98,0.12,0,0.24,-0.08,0.97,0,-0.33,-1.52,1.53,1]"
remote_url_nike = "https://antimatter15.com/splat/?url=https://media.reshot.ai/models/nike_next/model.splat#[0.95,0.16,-0.26,0,-0.16,0.99,0.01,0,0.26,0.03,0.97,0,0.01,-1.96,2.82,1]"

remote_urls = [remote_url_default, remote_url_truck, remote_url_stump, remote_url_garden, 
              remote_url_treehill, remote_url_bicycle, remote_url_plush, remote_url_nike]
remote_names = ['Default','Truck','Stump','Garden','Treehill','Bicycle','Plush','Nike']
remote_md = f"Online version: " + \
            " | ".join([f"[{name}]({url})" for name, url in zip(remote_names, remote_urls)])

instruction = """
<div style="padding: 10px; border: 1px solid #f0ad4e; border-radius: 8px; background-color: #fcf8e3;">
    ⚠️ <strong>Instruction:</strong> 
        Drag-and-drop .splat file into the GUI.
</div>
"""

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
"""


class ReusableTCPServer(TCPServer):
    allow_reuse_address = True  # Allow reuse after shutdown


def serve_local(host=HOST, port=PORT):

    os.chdir(str(root_path / "gausssplats"))
    handler = SimpleHTTPRequestHandler

    url = local_url.format(host=host, port=port)
    print(f"\nServing {local_path} at {url}")

    global httpd
    httpd = ReusableTCPServer((host, port), handler)
    httpd.serve_forever()


def serve_local_view(host=HOST, port=PORT):

    # Run server in background thread
    global httpd
    server_thread = Thread(target=serve_local, args=(host, port), daemon=True)
    server_thread.start()

    url = local_url.format(host=host, port=port)
    return iframe.format(url=url)


# Define UI settings & layout

def create_ui(min_width: int = 25, host = HOST, port = PORT):

    column_kwargs = dict(variant='panel', min_width=min_width)
    url = local_url.format(host=host, port=port)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 🧊 Gauss-Splat Visualization")
        gr.Markdown(remote_md)

        with gr.Row():
            vhost = gr.Textbox(value=host, interactive=False, label="Server Host", min_width=50)
            vport = gr.Number(value=port, precision=0, label="Server Port", min_width=50)
            with gr.Column():
                button = gr.Button(value="Run Server", variant="primary", min_width=50)
                gr.Markdown(instruction)

        display = gr.HTML(label="Gauss-Splat iFrame", show_label=True)
        button.click(fn=serve_local_view, inputs=[vhost, vport], outputs=[display])

    return gui, None


if __name__ == "__main__":
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000)

