import math
import numpy as np
import gradio as gr

from typing import List

import cv2
from PIL import Image
from PIL.Image import Image as ImageClass

from . import shared as ui
from .utils import gradget, fill_prompts_mv

from ..src import shared
from ..src.utils import clear_torch_cache
from ..src.models import infer_diff360_i2p, infer_mvdiff_i2p, generate_panoview


def wrap_imgference(
    view_01_prompt: str,
    view_02_prompt: str,
    view_03_prompt: str,
    view_04_prompt: str,
    view_05_prompt: str,
    view_06_prompt: str,
    view_07_prompt: str,
    view_08_prompt: str,
    positive_prompt: str,
    negative_prompt: str,
    image: ImageClass,
    img_size: int,
    strength: float,
    guidance_scale: float,
    control_scale: float,
    inference_step: int,
):
    assert shared.pipeline == 'img2pano'
    assert shared.model_name in ['mvdiffusion', 'diffusion360']
    assert shared.model_base is not None

    if shared.model_name == 'mvdiffusion':
        prompts_8vw = []
        for prompt in [view_01_prompt, view_02_prompt, view_03_prompt, view_04_prompt,
                       view_05_prompt, view_06_prompt, view_07_prompt, view_08_prompt]:
            prompts_8vw.append(prompt if isinstance(prompt, str) 
                                                and prompt != '' else None)
        assert any(prompts_8vw)
        prompts_8vw = fill_prompts_mv(prompts_8vw)

        if not isinstance(image, np.ndarray):
            image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_size, img_size), 
                            cv2.INTER_AREA if np.max(image.shape) > img_size \
                        else cv2.INTER_LINEAR)

        config = dict(
            images_mv=image, 
            image_size=img_size, 
            prompt=prompts_8vw, num_views=8,
            diffusion_steps=inference_step, 
            guidance_scale=guidance_scale, 
        )
        generated = infer_mvdiff_i2p(shared.model_base, **config)

        # Combine to panorama view
        pano_view = generate_panoview(generated)
        # pano_view = pano_view[540:-540] # remove white padding
        
        # Output Formatting --> Gallery
        outputs = [(Image.fromarray(generated[i]), f'view_{i}') for i in range(8)]
        outputs += [(Image.fromarray(pano_view), 'view_panorama')]
        outputs += [(Image.fromarray(image), 'view_original')]

    else:
        if isinstance(image, ImageClass) is False:
            image = Image.fromarray(image)

        config = dict(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=image, mask=None,
            image_size=img_size, 
            strength=strength,
            diffusion_steps=inference_step, 
            guidance_scale=guidance_scale, 
            controlnet_scale=control_scale,
        )
        generated = infer_diff360_i2p(shared.model_base, **config)
        
        # Output Formatting --> Gallery
        outputs = [(generated, 'pano_view')]

    clear_torch_cache()
    return outputs


def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    gallery_kwargs = dict(columns=8, show_share_button=True, 
                                    show_download_button=True,
                                    show_fullscreen_button=True)

    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 🗺️ Single-to-Multi-view (Panorama)")

        with gr.Row(variant="panel"):

            with gr.Column():
                ui.gradio['img2pano_image_sv'] = gr.Image(label='Single View')

                ui.gradio['img2pano_posprompt'] = gr.Textbox(label='Positive Prompt', value=shared.positive_prompt, lines=3, max_lines=7)
                ui.gradio['img2pano_negprompt'] = gr.Textbox(label='Negative Prompt', value=shared.negative_prompt, lines=3, max_lines=7)

                ui.gradio['img2pano_prompt_vw1'] = gr.Textbox(label='Prompt View 0°', placeholder='Describe the scene at 0°', lines=1, max_lines=3)
                ui.gradio['img2pano_prompt_guy'] = gr.Markdown(label='Prompt Guide', value="ℹ️ **Note**: Expand the panel below to insert view-specific prompt.")

                with gr.Accordion('Multi-view', open=False):
                    ui.gradio['img2pano_prompt_vw2'] = gr.Textbox(label='Prompt View 45°' , placeholder='Describe the scene at 45°' , lines=1, max_lines=3)
                    ui.gradio['img2pano_prompt_vw3'] = gr.Textbox(label='Prompt View 90°' , placeholder='Describe the scene at 90°' , lines=1, max_lines=3)
                    ui.gradio['img2pano_prompt_vw4'] = gr.Textbox(label='Prompt View 135°', placeholder='Describe the scene at 135°', lines=1, max_lines=3)
                    ui.gradio['img2pano_prompt_vw5'] = gr.Textbox(label='Prompt View 180°', placeholder='Describe the scene at 180°', lines=1, max_lines=3)
                    ui.gradio['img2pano_prompt_vw6'] = gr.Textbox(label='Prompt View 225°', placeholder='Describe the scene at 225°', lines=1, max_lines=3)
                    ui.gradio['img2pano_prompt_vw7'] = gr.Textbox(label='Prompt View 270°', placeholder='Describe the scene at 270°', lines=1, max_lines=3)
                    ui.gradio['img2pano_prompt_vw8'] = gr.Textbox(label='Prompt View 315°', placeholder='Describe the scene at 315°', lines=1, max_lines=3)

            with gr.Column():
                ui.gradio['img2pano_image_mv'] = gr.Gallery(label='Multi-view / Panorama-360', **gallery_kwargs)

                with gr.Row():
                    with gr.Column(scale=1, min_width=5):
                        gr.Markdown('')
                    with gr.Column(scale=3, min_width=25):
                        ui.gradio['img2pano_trigger'] = gr.Button(value=ui.symbols["trigger"], variant="primary")
                    with gr.Column(scale=1, min_width=5):
                        gr.Markdown('')

        with gr.Row(variant="panel"):
            with gr.Column():
                ui.gradio["img2pano_strength"] = gr.Slider(minimum=0.1, maximum=0.99, step=.01, value=0.9, label='Strength')
                ui.gradio["img2pano_guidance"] = gr.Slider(minimum=1.0, maximum=49.9, step=0.1, value=9.5, label='Guidance Scale')
                ui.gradio["img2pano_ctrl_scale"] = gr.Slider(minimum=.1, maximum=.99, step=.01, value=.95, label='Control Scale')

            with gr.Column():
                ui.gradio["img2pano_num_steps"] = gr.Slider(minimum=10, maximum=100, step=1, value=20, label='Diffusion Steps')
                ui.gradio["img2pano_img_size"] = gr.Slider(minimum=128, maximum=2048, step=16, value=768, label='Image Size')

        # Event Handle
        outputs = ui.gradio['img2pano_image_mv']
        inputs = gradget([f'img2pano_{x}' for x in \
                        ([f'prompt_vw{i}' for i in range(1, 9)] + \
                         ['posprompt','negprompt','image_sv','img_size',
                           'strength','guidance','ctrl_scale','num_steps'])])

        ui.gradio['img2pano_trigger'].click(fn=wrap_imgference, inputs=inputs, outputs=outputs)

        # Schedule
        def filter_promptbox():
            return \
                [gr.update(visible=(shared.model_name == "diffusion360")) for _ in range(2)] + \
                [gr.update(visible=(shared.model_name == "mvdiffusion")) for _ in range(9)]
        
        prompt_boxes = ['posprompt','negprompt']
        prompt_boxes += (['prompt_guy'] + [f'prompt_vw{i}' for i in range(1, 9)])
        prompt_boxes = gradget(*[f'img2pano_{x}' for x in prompt_boxes])

        timer = gr.Timer(value=10)
        timer.tick(fn=filter_promptbox, inputs=None, outputs=prompt_boxes)

    return gui


if __name__ == "__main__":
    gui= create_ui()
    gui.launch(server_name='localhost', server_port=8000)

