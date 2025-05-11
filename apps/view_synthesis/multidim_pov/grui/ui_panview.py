"""
Emojis: 🗺️
Reference: https://github.com/JeremyHeleine/Photo-Sphere-Viewer
"""
from pathlib import Path
import gradio as gr

import os
import threading
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer


root_path = Path(__file__).resolve().parent

PORT = 4321
local_path = root_path / "photo_sphere_viewer" / "index.html"
local_url = f"http://localhost:{PORT}/index.html"


class ReusableTCPServer(TCPServer):
    allow_reuse_address = True  # Allow reuse after shutdown


def serve_local_editor():

    os.chdir(str(root_path / "photo_sphere_viewer"))
    handler = SimpleHTTPRequestHandler

    global httpd
    httpd = ReusableTCPServer(("localhost", PORT), handler)
    print(f"\nServing {local_path} at {local_url}")
    httpd.serve_forever()


# Define UI settings & layout

def create_ui(min_width: int = 25):

    # Run server in background thread
    global httpd
    server_thread = threading.Thread(target=serve_local_editor, daemon=True)
    server_thread.start()

    # Gradio
    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("### 🗺️ Panorama Viewer")

        iframe = f"""<iframe id="pano360_iframe" src="{local_url}" width="100%" height="768"></iframe>"""
        gr.HTML(label="Pano360 iFrame", value=iframe, show_label=True)

    return gui, None


if __name__ == "__main__":

    # Run WebUI
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000)

    # Exit
    print("\nShutting down server...")
    httpd.shutdown()
    httpd.server_close()
    print("Server stopped.")

