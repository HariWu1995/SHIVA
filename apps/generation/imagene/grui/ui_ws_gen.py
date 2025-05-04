import gradio as gr

from . import shared as ui
from .utils import gradget, update_prompt, update_nrompt, extend_prompt, apply_styles

from ..src import shared
from ..src.utils import clear_torch_cache
from ..src.default import STYLES, POSITIVE_PROMPT, NEGATIVE_PROMPT
from ..src.pipelines import run_pipeline_generate as run_pipeline


def wrap_inference(
    prompt,
    nrompt,
    srompt,
    strength, 
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
        strength = strength, 
        guidance_scale = guidance, 
        num_inference_steps = num_steps, 
        output_type = 'pil',
    )

    clear_torch_cache()

    prompt = extend_prompt(prompt, srompt)
    generated = run_pipeline(shared.model, prompt, nrompt, **config)

    # Output Formatting --> Gallery
    outputs = [(img, f'gen_{i}') for i, img in enumerate(generated)]
    return outputs


# Define UI settings & layout

def create_ui():

    # NOTE: 
    # SD uses a process called tokenization to parse text prompts into cues. 
    # Maximum prompt length is 75 tokens, approximately 350 characters.

    with gr.Tab(label='Generation') as ui.gradio["tab_gen"]:
        with gr.Row():

            with gr.Column(scale=3):
                ui.gradio["tab_gen_prompt"] = gr.Textbox(label='Positive Prompt', max_lines=5, lines=3, max_length=200, placeholder="Description to include", value=POSITIVE_PROMPT)
                ui.gradio["tab_gen_nrompt"] = gr.Textbox(label='Negative Prompt', max_lines=2, lines=1, max_length=50, placeholder="Description to exclude", value=NEGATIVE_PROMPT)
                ui.gradio["tab_gen_styles"] = gr.Textbox(label='Stylized Prompt', max_lines=3, lines=1, max_length=100, placeholder="Description of styles")

                # prompt_trigger = gr.Button(value='Load prompt ✅')
                # nrompt_trigger = gr.Button(value='Load prompt ⛔')
                style_appender = gr.Dropdown(label='Preset Style Prompt', value=None, choices=STYLES, multiselect=True, max_choices=3)

            with gr.Column(scale=1):
                ui.gradio["tab_gen_strength"] = gr.Slider(minimum=.1, maximum=.99, step=.01, value=0.9, label='Strength')
                ui.gradio["tab_gen_guidance"] = gr.Slider(minimum=1., maximum=49, step=0.1, value=7.7, label='Guidance Scale')
                ui.gradio["tab_gen_num_steps"] = gr.Slider(minimum=10, maximum=100, step=1, value=30, label='Diffusion Steps')

            with gr.Column(scale=1):
                ui.gradio["tab_gen_width"] = gr.Slider(minimum=128, maximum=2048, step=16, value=512, label='Width')
                ui.gradio["tab_gen_height"] = gr.Slider(minimum=128, maximum=2048, step=16, value=512, label='Height')
                ui.gradio["tab_gen_num_outs"] = gr.Slider(minimum=1, maximum=100, step=1, value=1, label='Batch Size')

            with gr.Column(scale=5):
                ui.gradio["tab_gen_genlery"] = gr.Gallery(label="Generated images", show_label=True, elem_id="gallery", 
                                                         columns=[4], rows=[1], object_fit="contain", height="auto")
                ui.gradio["tab_gen_button"] = gr.Button(value="Generate", variant='primary')

        # Event handlers
        # prompt_trigger.click(fn=update_prompt, inputs=gradget("tab_gen_prompt"), outputs=gradget("tab_gen_prompt"))
        # nrompt_trigger.click(fn=update_nrompt, inputs=gradget("tab_gen_nrompt"), outputs=gradget("tab_gen_nrompt"))
        style_appender.select(fn=apply_styles, inputs=gradget("tab_gen_styles") + [style_appender], 
                                              outputs=gradget("tab_gen_styles"))

        ui.gradio["tab_gen_button"].click(
                fn=wrap_inference,
             inputs=gradget("tab_gen_prompt","tab_gen_nrompt","tab_gen_styles",
                            "tab_gen_strength","tab_gen_guidance","tab_gen_num_steps",
                            "tab_gen_num_outs","tab_gen_height","tab_gen_width"),
            outputs=gradget("tab_gen_genlery"),
        )

        # Example
        # ui.gradio["tab_gen_examples"] = gr.Examples()

    return ui.gradio["tab_gen"]


if __name__ == "__main__":
    gui = create_ui()
    gui.launch(server_name='localhost', server_port=8000)

