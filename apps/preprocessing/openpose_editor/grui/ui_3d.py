"""
Emojis: 🤸‍♀️🤸‍♂️𖠋
"""
from pathlib import Path
import gradio as gr

import os
import threading
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer


root_path = Path(__file__).resolve().parents[0]

PORT = 5003
local_path = root_path / "pose3d" / "index.html"
local_url = f"http://localhost:{PORT}/index.html"
remote_url_1 = "https://zhuyu1997.github.io/open-pose-editor/"
remote_url_2 = "https://openposeai.com/"


class ReusableTCPServer(TCPServer):
    allow_reuse_address = True  # Allow reuse after shutdown


def serve_local_editor():

    os.chdir(str(root_path / "pose3d"))
    handler = SimpleHTTPRequestHandler

    global httpd
    httpd = ReusableTCPServer(("localhost", PORT), handler)
    print(f"\nServing {local_path} at {local_url}")
    httpd.serve_forever()


# Define UI settings & layout

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 𖠋 Human-Pose Keypoints Editor")

        iframe = f"""<iframe id="openpose3d_iframe" src="{local_url}" width="100%" height="768"></iframe>"""
        gr.HTML(label="OpenPose3D iFrame", value=iframe, show_label=True)
        gr.Markdown(f"Online version: [3D Openpose Editor]({remote_url_1}) / [OpenPoseAI.com]({remote_url_2})")

    return gui, None


if __name__ == "__main__":

    # Run server in background thread
    global httpd
    server_thread = threading.Thread(target=serve_local_editor, daemon=True)
    server_thread.start()

    # Run WebUI
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000)

    # Exit
    print("\nShutting down server...")
    httpd.shutdown()
    httpd.server_close()
    print("Server stopped.")
