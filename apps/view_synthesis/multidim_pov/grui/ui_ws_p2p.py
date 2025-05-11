import math
import pandas as pd
import gradio as gr

from typing import List
from PIL import Image
from PIL.Image import Image as ImageClass

from . import shared as ui
from .utils import gradget, fill_prompts_mv

from ..src import shared
from ..src.utils import clear_torch_cache
from ..src.models import infer_diff360_p2p


def wrap_imghancement(
    positive_prompt: str,
    negative_prompt: str,
    image: ImageClass,
    upscale: int,
    strength: float,
    guidance_scale: float,
    control_scale: float,
    inference_step: int,
):
    # assert shared.model_name == 'diffusion360'
    assert shared.model_refine is not None

    if isinstance(image, ImageClass) is False:
        image = Image.fromarray(image)

    # Resize before refine
    w, h = image.size
    w_fit = math.ceil(w / 768) * 768
    h_fit = math.ceil(h / 768) * 768
    image = image.resize((w_fit, h_fit), Image.LANCZOS)

    config = dict(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=image, 
        upscale=upscale,
        strength=strength,
        diffusion_steps=inference_step, 
        guidance_scale=guidance_scale, 
        controlnet_scale=control_scale,
    )

    generated = infer_diff360_p2p(shared.model_refine, **config)
    clear_torch_cache()
    return generated


def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    gallery_kwargs = dict(columns=8, show_share_button=True, 
                                    show_download_button=True,
                                    show_fullscreen_button=True)

    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 🗺️ Panorama Enhancement")

        with gr.Row(variant="panel"):
            with gr.Column():
                ui.gradio['pano2pano_image_in'] = gr.Image(label='Original')
            with gr.Column():
                ui.gradio['pano2pano_image_out'] = gr.Image(label='Enhanced')
                with gr.Row():
                    with gr.Column(scale=1, min_width=5):
                        gr.Markdown('')
                    with gr.Column(scale=3, min_width=25):
                        ui.gradio['pano2pano_trigger'] = gr.Button(value=ui.symbols["trigger"], variant="primary")
                    with gr.Column(scale=1, min_width=5):
                        gr.Markdown('')


        with gr.Row(variant="panel"):

            with gr.Column():
                ui.gradio['pano2pano_posprompt'] = gr.Textbox(label='Positive Prompt', value=shared.positive_prompt, lines=5, max_lines=7)
                ui.gradio['pano2pano_negprompt'] = gr.Textbox(label='Negative Prompt', value=shared.negative_prompt, lines=5, max_lines=7)

            with gr.Column():
                ui.gradio["pano2pano_strength"] = gr.Slider(minimum=0.1, maximum=0.99, step=.01, value=0.5, label='Strength')
                ui.gradio["pano2pano_guidance"] = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, value=7.0, label='Guidance Scale')
                ui.gradio["pano2pano_ctrl_scale"] = gr.Slider(minimum=.1, maximum=.99, step=.01, value=.95, label='Control Scale')
                ui.gradio["pano2pano_n_steps"] = gr.Slider(minimum=1, maximum=20, step=1, value=10, label='Diffusion Steps')
                ui.gradio["pano2pano_upscale"] = gr.Slider(minimum=1, maximum=8, step=1, value=2, label='Upscale')

        # Event Handle
        outputs = ui.gradio['pano2pano_image_out']
        inputs = gradget([f'pano2pano_{x}' for x in \
                          ['posprompt','negprompt','image_in','upscale',
                           'strength','guidance','ctrl_scale','n_steps']])

        ui.gradio['pano2pano_trigger'].click(fn=wrap_imghancement, inputs=inputs, outputs=outputs)

    return gui


if __name__ == "__main__":
    gui= create_ui()
    gui.launch(server_name='localhost', server_port=8000)

