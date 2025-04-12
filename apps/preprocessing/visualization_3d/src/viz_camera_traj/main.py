from pathlib import Path
import threading
import time
import httpx

import numpy as np
import gradio as gr
import viser

from .utils import set_bg_color
from .render import Renderer
from .examples import EXAMPLE_MAP


SERVERS = {}
ABORT_EVENTS = {}

DUST3R_CKPT_PATH = Path(__file__).resolve().parents[5] / "checkpoints" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"


# Make sure that gradio uses dark theme.
_APP_JS = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
    }
}
"""


def start_server_and_abort_event(
    request: gr.Request, 
    host: str = "localhost",
    port: int = 1234,
):
    server = viser.ViserServer(host=host, port=port)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        # Force dark mode that blends well with gradio's dark theme.
        client.gui.configure_theme(dark_mode=True, show_share_button=False, control_layout="collapsible")
        set_bg_color(client)

    server_url = f"http://{server.get_host()}:{server.get_port()}"
    print(f"Starting server @ {server_url}")
    SERVERS[request.session_hash] = server
    if server_url is None:
        raise gr.Error("Failed to get a viewport URL. Please check your network connection.")
    
    # Give it enough time to start.
    time.sleep(1)

    ABORT_EVENTS[request.session_hash] = threading.Event()

    return (
        Renderer(server, model_path=DUST3R_CKPT_PATH),
        gr.HTML(
            f'<iframe src="{server_url}" style="display: block; margin: auto; width: 100%; height: max(60vh, 600px);" frameborder="0"></iframe>',
            container=True,
        ),
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
    viser_port: int = 1234,
    server_port: int = 8080, 
    share: bool = False,
):
    with gr.Blocks(js=_APP_JS) as app:
        renderer = gr.State()
        session = gr.State()

        render_btn = gr.Button("Render video", interactive=False, render=False)
        viewport = gr.HTML(container=True, render=False)

        gr.Timer(0.1).tick(
            lambda renderer: gr.update(
                interactive=renderer is not None
                and renderer.gui_state is not None
                and renderer.gui_state.camera_traj_list is not None
            ),
            inputs=[renderer],
            outputs=[render_btn],
        )

        with gr.Row():
            viewport.render()
        
        with gr.Row():
            with gr.Column():
                
                with gr.Group():
                    # Initially disable the Preprocess Images button until images are selected.
                    preprocess_btn = gr.Button("Preprocess images", interactive=False)
                    preprocess_pbar = gr.Textbox(label="Progress", interactive=False, visible=False)
                
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

                    chunk_strategy = gr.Dropdown(
                        value="interp",
                        choices=["interp", "interp-gt"],
                        label="Chunk strategy",
                        render=False,
                    )

                    with gr.Row():
                        example_imgs_backer = gr.Button("Go back", visible=False)
                        example_imgs_confirmer = gr.Button("Confirm", visible=False)

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
                            preprocess_btn
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
                        outputs=[preprocess_btn],
                    )

                    preprocess_btn.click(
                        lambda r, *args: r.preprocess(*args),
                        inputs=[renderer, input_imgs],
                        outputs=[preprocessed, preprocess_pbar, chunk_strategy],
                        show_progress_on=[preprocess_pbar],
                        concurrency_id="gpu_queue",
                    )

                    preprocess_btn.click(
                        lambda: gr.update(visible=True),
                        outputs=[preprocess_pbar],
                    )

                    preprocessed.change(
                        lambda r, *args: r.visualize_scene(*args),
                        inputs=[renderer, preprocessed],
                    )

                with gr.Row():
                    seed = gr.Number(value=23, label="Random seed")
                    chunk_strategy.render()
                    cfg = gr.Slider(1.0, 7.0, value=3.0, label="CFG value")

                with gr.Row():
                    camera_scale = gr.Slider(0.1, 15.0, value=2.0, label="Camera scale (useful for single-view input)")

                with gr.Group():
                    output_data_dir = gr.Textbox(label="Output data directory")
                    output_data_btn = gr.Button(value="Export output data")

                output_data_btn.click(
                    lambda r, *args: r.export_output_data(*args),
                    inputs=[renderer, preprocessed, output_data_dir],
                )

            with gr.Column():
                with gr.Group():
                    abort_btn = gr.Button("Abort rendering", visible=False)
                    render_btn.render()
                    render_progress = gr.Textbox(label="", visible=False, interactive=False)
                
                output_video = gr.Video(label="Output", interactive=False, autoplay=True, loop=True)
                
                render_btn.click(
                    fn=lambda r, *args: (yield from r.render(*args)),
                    inputs=[renderer, preprocessed, session, seed, chunk_strategy, 
                            cfg, gr.State(), gr.State(), gr.State(), camera_scale],
                    outputs=[output_video, render_btn, abort_btn, render_progress],
                    show_progress_on=[render_progress],
                    concurrency_id="gpu_queue",
                )
    
                render_btn.click(
                    fn=lambda: [gr.update(visible=False), 
                                gr.update(visible=True),
                                gr.update(visible=True)],
                    outputs=[render_btn, abort_btn, render_progress],
                )

                abort_btn.click(set_abort_event)

        # Register the session initialization and cleanup functions.
        app.load(fn=start_server_and_abort_event, outputs=[renderer, viewport, session])
        app.unload(fn=stop_server_and_abort_event)

    app.queue(max_size=5).launch(
        share=share,
        server_port=server_port,
        show_error=True,
        # allowed_paths=[WORK_DIR],
        # Badget rendering will be broken otherwise.
        ssr_mode=False,
    )


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
