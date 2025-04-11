"""
Emojis: 🤸‍♀️🤸‍♂️𖠋
"""
from pathlib import Path
import gradio as gr
import uvicorn
import threading

from .pose2d.scripts.openpose_editor import FastAPI, mount_openpose_api


root_path = Path(__file__).resolve().parents[0]

PORT = 5002
local_url = f"http://localhost:{PORT}/openpose_editor"
remote_url = "https://huchenlei.github.io/sd-webui-openpose-editor/"


def serve_local_editor():

    print(f"\nServing at {local_url}")
    app = FastAPI()
    mount_openpose_api(app)
    uvicorn.run(app, host="localhost", port=PORT, reload=False)


# Define UI settings & layout

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 𖠋 Human-Pose Keypoints Editor")

        iframe = f"""<iframe id="openpose2d_iframe" src="{local_url}" width="100%" height="768"></iframe>"""
        gr.HTML(label="OpenPose2D iFrame", value=iframe, show_label=True)
        gr.Markdown(f"Online version: [2D Openpose Editor]({remote_url})")

    return gui, None


if __name__ == "__main__":

    # Run server in background thread
    server_thread = threading.Thread(target=serve_local_editor, daemon=True)
    server_thread.start()

    # Run WebUI
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000)

