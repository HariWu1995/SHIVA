import gradio as gr

from . import shared as ui
from .utils import gradget, update_prompt, update_nrompt, extend_prompt, apply_styles

from ..src import shared
from ..src.utils import clear_torch_cache
from ..src.default import STYLES, EXAMPLES
from ..src.pipelines import run_pipeline_inpaint as run_pipeline
from ..src.pipelines.utils import preprocess_brushnet


def wrap_inference(
    prompt,
    nrompt,
    image,
    mask,
    strength, 
    guidance, 
    brush_scale,
    num_steps, 
    batch_size,
    height,
    width,
):
    config = dict(
        height = height, 
        width = width,
        batch_size = batch_size,
        strength = strength, 
        guidance_scale = guidance, 
        brushnet_scale = float(brush_scale) if shared.brush_name else None,
        num_inference_steps = num_steps, 
        output_type = 'pil',
    )

    clear_torch_cache()
    if shared.brush_name:
        image, mask = preprocess_brushnet(image, mask)
    generated = run_pipeline(shared.model, image, mask, prompt, nrompt, **config)

    # Output Formatting --> Gallery
    outputs = [(img, f'gen_{i}') for i, img in enumerate(generated)]
    return outputs


# Define UI settings & layout

def create_ui():

    # NOTE: 
    # SD uses a process called tokenization to parse text prompts into cues. 
    # Maximum prompt length is 75 tokens, approximately 350 characters.

    with gr.Tab(label='Fulfill') as ui.gradio["tab_fill"]:
        with gr.Row():

            with gr.Column():
                ui.gradio["tab_fill_image"] = gr.Image(label="Image")
                ui.gradio["tab_fill_prompt"] = gr.Textbox(label='Positive Prompt', max_lines=5, lines=2, max_length=200, placeholder="Description to include")

                ui.gradio["tab_fill_strength"] = gr.Slider(minimum=.1, maximum=.99, step=.01, value=0.9, label='Strength')
                ui.gradio["tab_fill_guidance"] = gr.Slider(minimum=1., maximum=49, step=0.1, value=7.7, label='Guidance Scale')
                ui.gradio["tab_fill_brushnet"] = gr.Slider(minimum=.1, maximum=2., step=0.1, value=1.0, label='Brushnet Scale')
                ui.gradio["tab_fill_num_steps"] = gr.Slider(minimum=10, maximum=100, step=1, value=30, label='Diffusion Steps')

            with gr.Column():
                ui.gradio["tab_fill_mask"] = gr.Image(label="Mask")
                ui.gradio["tab_fill_nrompt"] = gr.Textbox(label='Negative Prompt', max_lines=3, lines=2, max_length=50, placeholder="Description to exclude")

                ui.gradio["tab_fill_width"] = gr.Slider(minimum=128, maximum=2048, step=16, value=512, label='Width')
                ui.gradio["tab_fill_height"] = gr.Slider(minimum=128, maximum=2048, step=16, value=512, label='Height')
                ui.gradio["tab_fill_num_outs"] = gr.Slider(minimum=1, maximum=100, step=1, value=1, label='Batch Size')

            with gr.Column():
                ui.gradio["tab_fill_fillery"] = gr.Gallery(label="Completed images", show_label=True, elem_id="gallery", 
                                                        columns=[4], rows=[1], object_fit="contain", height="auto")
                ui.gradio["tab_fill_button"] = gr.Button(value="Complete", variant='primary')

        ## Event handlers
        ui.gradio["tab_fill_button"].click(
                fn=wrap_inference,
             inputs=gradget("tab_fill_prompt","tab_fill_nrompt",
                            "tab_fill_image","tab_fill_mask",
                            "tab_fill_strength","tab_fill_guidance","tab_fill_brushnet",
                            "tab_fill_num_steps","tab_fill_num_outs",
                            "tab_fill_height","tab_fill_width"),
            outputs=gradget("tab_fill_fillery"),
        )

    return ui.gradio["tab_fill"]


if __name__ == "__main__":
    gui = create_ui()
    gui.launch(server_name='localhost', server_port=8000)

