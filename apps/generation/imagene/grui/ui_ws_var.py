import gradio as gr

from . import shared as ui
from .utils import gradget, update_prompt, update_nrompt, extend_prompt, apply_styles

from ..src import shared
from ..src.utils import clear_torch_cache
from ..src.default import STYLES, EXAMPLES
from ..src.pipelines import run_pipeline_variate as run_pipeline


def wrap_inference(
    prompt,
    nrompt,
    template,
    adaption, 
    guidance, 
    num_steps, 
    batch_size,
    height,
    width,
):
    config = dict(
        height = height, 
        width = width,
        batch_size = batch_size,
        adapt_scale = adaption, 
        guidance_scale = guidance, 
        num_inference_steps = num_steps, 
        output_type = 'pil',
    )

    clear_torch_cache()
    template = template.convert("RGB")
    generated = run_pipeline(shared.model, template, prompt, nrompt, **config)

    # Output Formatting --> Gallery
    outputs = [(img, f'gen_{i}') for i, img in enumerate(generated)]
    return outputs


# Define UI settings & layout

def create_ui():

    # NOTE: 
    # SD uses a process called tokenization to parse text prompts into cues. 
    # Maximum prompt length is 75 tokens, approximately 350 characters.

    with gr.Tab(label='Variation') as ui.gradio["tab_var"]:
        with gr.Row():

            with gr.Column():
                ui.gradio["tab_var_target"] = gr.Image(label="Image Template")
                ui.gradio["tab_var_prompt"] = gr.Textbox(label='Positive Prompt', max_lines=3, lines=2, max_length=200, placeholder="Description to include")
                ui.gradio["tab_var_nrompt"] = gr.Textbox(label='Negative Prompt', max_lines=1, lines=1, max_length=50, placeholder="Description to exclude")

                ui.gradio["tab_var_adaption"] = gr.Slider(minimum=.1, maximum=2., step=0.1, value=.55, label='Adaption Scale')
                ui.gradio["tab_var_guidance"] = gr.Slider(minimum=1., maximum=49, step=0.1, value=7.7, label='Guidance Scale')
                ui.gradio["tab_var_num_steps"] = gr.Slider(minimum=10, maximum=100, step=1, value=30, label='Diffusion Steps')

            with gr.Column():
                ui.gradio["tab_var_varlery"] = gr.Gallery(label="Variated images", show_label=True, elem_id="gallery", 
                                                        columns=[4], rows=[1], object_fit="contain", height="auto")
                ui.gradio["tab_var_button"] = gr.Button(value="Variate", variant='primary')
                ui.gradio["tab_var_width"] = gr.Slider(minimum=128, maximum=2048, step=16, value=512, label='Width')
                ui.gradio["tab_var_height"] = gr.Slider(minimum=128, maximum=2048, step=16, value=512, label='Height')
                ui.gradio["tab_var_num_outs"] = gr.Slider(minimum=1, maximum=100, step=1, value=1, label='Batch Size')

        ## Event handlers
        ui.gradio["tab_var_button"].click(
                fn=wrap_inference,
             inputs=gradget("tab_var_prompt","tab_var_nrompt","tab_var_target",
                            "tab_var_adaption","tab_var_guidance","tab_var_num_steps",
                            "tab_var_num_outs","tab_var_height","tab_var_width"),
            outputs=gradget("tab_var_varlery"),
        )

    return ui.gradio["tab_var"]


if __name__ == "__main__":
    gui = create_ui()
    gui.launch(server_name='localhost', server_port=8000)

