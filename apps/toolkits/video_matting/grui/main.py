import os
import gradio as gr

from pathlib import Path
from ..src import all_models, default_model, inference


def wrap_inference(video_in, model_variant, output_dir):
    video_name = os.path.normpath(video_in).split(os.sep)[-1]
    video_composed   = os.path.join(output_dir, video_name[:-4] + '_composed.mp4')
    video_foreground = os.path.join(output_dir, video_name[:-4] + '_foreground.mp4')
    video_alpha_mask = os.path.join(output_dir, video_name[:-4] + '_alpha.mp4')
    inference(
        model_variant = model_variant,
        input_source = video_in,
        output_type = "video",
        output_composition = video_composed,
        output_foreground = video_foreground,
        output_alpha_mask = video_alpha_mask,
    )
    return video_composed, video_alpha_mask, video_foreground


# Define UI settings & layout

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    temp_dir = Path(__file__).parents[4] / 'temp'
    if os.path.isdir(temp_dir) is False:
        os.makedirs(temp_dir)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 🖼 Video Matting (Decomposition)")

        with gr.Row():
            video_in = gr.Video(label='Input Video')
            video_out = gr.Video(label='Matted Video')

        with gr.Row():
            with gr.Column(scale=1):
                run_button = gr.Button(value="Run", variant='primary')
            with gr.Column(scale=5):
                gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=1):
                model_var = gr.Dropdown(all_models, value=default_model, label="Select Model")
            with gr.Column(scale=5):
                output_dir = gr.Textbox(value=temp_dir, interactive=True, label="Output Directory")

        with gr.Row():
            video_mask = gr.Video(label='Mask')
            video_fore = gr.Video(label='Foreground')

        # Functionality
        run_button.click(fn=wrap_inference, 
                     inputs=[video_in, model_var, output_dir], 
                    outputs=[video_out, video_mask, video_fore])

    return gui, [video_out, video_mask]


if __name__ == "__main__":
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000)

