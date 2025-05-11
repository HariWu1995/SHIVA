import math
import gradio as gr

from typing import List
from PIL import Image
from PIL.Image import Image as ImageClass

from . import shared as ui
from .utils import gradget

from ..src import shared
from ..src.utils import clear_torch_cache
from ..src.models import infer_diff360_i2p, refine_diff360_i2p, \
                          infer_mvdiff_i2p, generate_panoview


def wrap_imghancement(
    positive_prompt: str,
    negative_prompt: str,
    images: List[ImageClass],
    img_size: int,
    strength: float,
    guidance_scale: float,
    control_scale: float,
    inference_step: int,
):
    assert shared.model_name == 'diffusion360'
    assert shared.model is not None

    # Resize before refine
    img_size = math.ceil(img_size  / 768) * 768
    height = img_size
    width = img_size * 2

    image = images[-1][0]
    image = image.resize((width, height), Image.LANCZOS)

    config = dict(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=image, 
        upscale=-1,
        strength=strength,
        diffusion_steps=inference_step, 
        guidance_scale=guidance_scale, 
        controlnet_scale=control_scale,
    )
    generated = refine_diff360_i2p(shared.model, **config)
    
    # Output Formatting --> Gallery
    outputs = images + [(generated, f'{images[-1][1]}_x2')]

    return outputs


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
    assert shared.model is not None

    if shared.model_name == 'mvdiffusion':
        prompts_8vw = []
        for prompt in [view_01_prompt, view_02_prompt, view_03_prompt, view_04_prompt,
                       view_05_prompt, view_06_prompt, view_07_prompt, view_08_prompt]:
            prompts_8vw.append(prompt if isinstance(prompt, str) and prompt != '' 
                                    else None)
        assert any(prompts_8vw)
        prompts_8vw = fill_prompts_mv(prompts_8vw)

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        config = dict(
            images_mv=image, 
            image_size=img_size, 
            prompt=prompts_8vw, num_views=8,
            diffusion_steps=inference_step, 
            guidance_scale=guidance_scale, 
        )
        generated = infer_mvdiff_i2p(shared.model, **config)

        # Combine to panorama view
        pano_view = generate_panoview(generated)
        # pano_view = pano_view[540:-540]
        
        # Output Formatting --> Gallery
        outputs = [(Image.fromarray(generated[i]), f'gen_{i}') for i in range(8)]
        outputs += [(Image.fromarray(pano_view), 'pano_view')]

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
        generated = infer_diff360_i2p(shared.model, **config)
        
        # Output Formatting --> Gallery
        outputs = [(generated, 'pano_view')]

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
                gr.Markdown("ℹ️ **Note**: **Refine** is only available with **Diffusion-360** model.")
                gr.Markdown("⚠️ **Note**: Increase **image size** and decrease other params before clicking **Refine**.")

                with gr.Row():
                    with gr.Column(scale=1, min_width=5):
                        gr.Markdown('')
                    with gr.Column(scale=3, min_width=25):
                        ui.gradio['img2pano_trigger'] = gr.Button(value=ui.symbols["trigger"], variant="primary")
                    with gr.Column(scale=3, min_width=25):
                        ui.gradio['img2pano_refiner'] = gr.Button(value=ui.symbols["refiner"], variant="secondary", interactive=False)
                    with gr.Column(scale=1, min_width=5):
                        gr.Markdown('')

        with gr.Row(variant="panel"):
            with gr.Column():
                ui.gradio["img2pano_strength"] = gr.Slider(minimum=.1, maximum=.99, step=.01, value=0.9, label='Strength')
                ui.gradio["img2pano_guidance"] = gr.Slider(minimum=1., maximum=49, step=0.1, value=9.5, label='Guidance Scale')
                ui.gradio["img2pano_ctrl_scale"] = gr.Slider(minimum=.1, maximum=.99, step=.01, value=.95, label='Control Scale')
                ui.gradio["img2pano_num_steps"] = gr.Slider(minimum=10, maximum=100, step=1, value=20, label='Diffusion Steps')

            with gr.Column():
                ui.gradio["img2pano_img_size"] = gr.Slider(minimum=128, maximum=2048, step=16, value=768, label='Image Size')

        # Event Handle
        gen_inputs = gradget([f'img2pano_{x}' for x in \
                            ([f'prompt_vw{i}' for i in range(1, 9)] + \
                            ['posprompt','negprompt','image_sv','img_size',
                            'strength','guidance','ctrl_scale','num_steps'])])

        re_inputs = gradget([f'img2pano_{x}' for x in [
                            'posprompt','negprompt','image_mv','img_size',
                            'strength','guidance','ctrl_scale','num_steps']])

        outputs = ui.gradio['img2pano_image_mv']

        ui.gradio['img2pano_trigger'].click(fn=wrap_imgference, inputs=gen_inputs, outputs=outputs)
        ui.gradio['img2pano_refiner'].click(fn=wrap_imghancement, inputs=re_inputs, outputs=outputs)

        # Schedule
        def verify_refiner():
            return gr.update(interactive=(shared.model_name == "diffusion360"))

        timer_1 = gr.Timer(value=1.0)
        timer_1.tick(fn=verify_refiner, inputs=None, outputs=ui.gradio['img2pano_refiner'])

        def filter_promptbox():
            return \
                [gr.update(visible=(shared.model_name == "diffusion360")) for _ in range(2)] + \
                [gr.update(visible=(shared.model_name == "mvdiffusion")) for _ in range(9)]
        
        prompt_boxes = ['posprompt','negprompt']
        prompt_boxes += (['prompt_guy'] + [f'prompt_vw{i}' for i in range(1, 9)])
        prompt_boxes = gradget(*[f'img2pano_{x}' for x in prompt_boxes])

        timer_2 = gr.Timer(value=10)
        timer_2.tick(fn=filter_promptbox, inputs=None, outputs=prompt_boxes)

    return gui


if __name__ == "__main__":
    gui= create_ui()
    gui.launch(server_name='localhost', server_port=8000)

