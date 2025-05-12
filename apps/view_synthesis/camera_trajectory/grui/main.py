"""
Stable Virtual Camera
    https://github.com/Stability-AI/stable-virtual-camera/blob/main/demo_gr.py
"""
import time
import httpx
import threading

from pathlib import Path
from glob import glob

import numpy as np
import gradio as gr
import viser

from ..src.utils import set_bg_color
from .render import Renderer
from .utils import _APP_JS, USER_GUIDE, attention_catcher, find_available_ports


ROOT_DIR = Path(__file__).resolve().parents[4]

SAVE_DIR = ROOT_DIR / "temp"
SAMPLE_DIR = ROOT_DIR / "_samples" / "camera"

SERVERS = {}
ABORT_EVENTS = {}

EXAMPLE_MAP = []
for ex in  ['garden-4_*.jpg','telebooth-2_*.jpg','vgg-lab-4_*.png','backyard-7_*.jpg']:
    ex_mv = sorted(glob(str(SAMPLE_DIR / ex)))
    EXAMPLE_MAP.append((ex_mv[0], ex_mv))


def start_server_and_abort_event(
    host: str,
    port: int,
    request: gr.Request, 
):
    server = viser.ViserServer(host=host, port=port)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        # Force dark mode that blends well with gradio's dark theme.
        client.gui.configure_theme(dark_mode=True, show_share_button=False, control_layout="collapsible")
        set_bg_color(client)

    server_url = f"http://{server.get_host()}:{server.get_port()}"
    if server_url is None:
        raise gr.Error("Failed to get a viewport URL. Please check your network connection.")
    print(f"\n\nStarting Viser server @ {server_url}")
    
    # Give it enough time to start.
    time.sleep(3)

    SERVERS[request.session_hash] = server
    ABORT_EVENTS[request.session_hash] = threading.Event()

    iframe = f'<iframe src="{server_url}" style="display: block; margin: auto; width: 100%; height: max(60vh, 600px);" frameborder="0"></iframe>'

    return (
        Renderer(server),
        gr.HTML(iframe, container=True),
        request.session_hash,
    )


def stop_server_and_abort_event(request: gr.Request):
    if request.session_hash in SERVERS:
        print(f"Stopping server {request.session_hash}")
        server = SERVERS.pop(request.session_hash)
        server.stop()

    if request.session_hash in ABORT_EVENTS:
        print(f"Setting abort event {request.session_hash}")
        ABORT_EVENTS[request.session_hash].set()
        # Give it enough time to abort jobs.
        time.sleep(5)
        ABORT_EVENTS.pop(request.session_hash)


def set_abort_event(request: gr.Request):
    if request.session_hash in ABORT_EVENTS:
        print(f"Setting abort event {request.session_hash}")
        ABORT_EVENTS[request.session_hash].set()


def get_advance_examples(selection: gr.SelectData):
    index = selection.index
    return (
        gr.Gallery(EXAMPLE_MAP[index][1], visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.Gallery(visible=False),
    )


def main(
    server_port: int = 8080, 
    share: bool = False,
):
    with gr.Blocks(js=_APP_JS) as app:

        gr.Markdown("## 🎥 3D Camera Trajectory Setup")

        renderer = gr.State()
        process_btn = gr.Button("Process multi-view", interactive=False, variant="primary", render=False)
        process_pbar = gr.Textbox(label="Progress", interactive=False, visible=False, render=False)

        all_ports = find_available_ports(count=5)
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row(variant="panel"):
                    viser_host = gr.Textbox(value="localhost", interactive=False, label="Viser Host")
                    viser_port = gr.Dropdown(value=all_ports[0], choices=all_ports, label="Viser Port")
                    session = gr.Textbox(label="Session Hash", interactive=False)
            with gr.Column(scale=1, min_width=50):
                gr.Markdown(attention_catcher)
                viser = gr.Button("Run Viser", variant="primary")

        with gr.Accordion(label="🛠️ User Guide", open=False):
            gr.Markdown(USER_GUIDE)
        
        with gr.Row():
            viewport = gr.HTML(container=True, render=True)
        
        with gr.Row():

            with gr.Column(scale=3):
                
                with gr.Group():
                    input_imgs = gr.Gallery(label="Input", columns=4, interactive=True, height=200)

                    # Define example images 
                    # (gradio doesn't support variable length examples so we need to hack it).
                    example_imgs = gr.Gallery(
                        [e[0] for e in EXAMPLE_MAP],
                        allow_preview=False,
                        preview=False,
                        label="Example",
                        columns=20,
                        rows=1,
                        height=115,
                    )

                    example_imgs_expander = gr.Gallery(
                        visible=False,
                        interactive=False,
                        label="Example",
                        preview=True,
                        columns=20,
                        rows=1,
                    )

                    with gr.Row():
                        example_imgs_backer = gr.Button("Go back", visible=False)
                        example_imgs_confirmer = gr.Button("Confirm", visible=False, variant="primary")

                    example_imgs.select(
                        get_advance_examples,
                        outputs=[
                            example_imgs_expander,
                            example_imgs_confirmer,
                            example_imgs_backer,
                            example_imgs,
                        ],
                    )

                    example_imgs_confirmer.click(
                        lambda x: (
                            x,
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=True),
                            gr.update(interactive=bool(x))
                        ),
                        inputs=[example_imgs_expander],
                        outputs=[
                            input_imgs,
                            example_imgs_expander,
                            example_imgs_confirmer,
                            example_imgs_backer,
                            example_imgs,
                            process_btn,
                        ],
                    )

                    example_imgs_backer.click(
                        lambda: (
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=True),
                        ),
                        outputs=[
                            example_imgs_expander,
                            example_imgs_confirmer,
                            example_imgs_backer,
                            example_imgs,
                        ],
                    )
    
                    preprocessed = gr.State()

                    # Enable the Preprocess Images button when images are uploaded
                    input_imgs.change(
                        lambda imgs: gr.update(interactive=bool(imgs)),
                        inputs=[input_imgs],
                        outputs=[process_btn],
                    )

                    process_btn.click(
                        lambda _renderer, *args: [_renderer.preprocess(*args), gr.update(visible=False)],
                        inputs=[renderer, input_imgs],
                        outputs=[preprocessed, process_pbar],
                        show_progress_on=[process_pbar],
                        concurrency_id="gpu_queue",
                    )

                    process_btn.click(
                        lambda: gr.update(visible=True),
                        outputs=[process_pbar],
                    )

                    preprocessed.change(
                        lambda _renderer, *args: _renderer.visualize_scene(*args),
                        inputs=[renderer, preprocessed],
                    )

            with gr.Column(scale=1):
                
                with gr.Group():
                    # Initially disable the Process button until images are selected.
                    process_btn.render()
                    process_pbar.render()

                with gr.Group():
                    save_dir = gr.Textbox(value=str(SAVE_DIR), label="Output data directory")
                    save_btn = gr.Button(value="Save data")

                session.change(
                    fn = lambda x: str(SAVE_DIR / str(x)),
                    inputs = [session],
                    outputs = [save_dir],
                )

                save_btn.click(
                    lambda _renderer, *args: _renderer.export_output_data(*args),
                    inputs=[renderer, preprocessed, save_dir],
                )
                
        # Register the session initialization and cleanup functions.
        viser.click(fn=start_server_and_abort_event, inputs=[viser_host, viser_port], outputs=[renderer, viewport, session])
        app.unload(fn=stop_server_and_abort_event)

    app.queue(max_size=5).launch(
        share=share,
        server_port=server_port,
        show_error=True,
    )


if __name__ == "__main__":
    import tyro
    tyro.cli(main)

