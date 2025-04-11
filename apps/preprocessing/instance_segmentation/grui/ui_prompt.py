"""
Reference:
    https://www.gradio.app/docs/gradio/annotatedimage
    https://www.gradio.app/docs/gradio/imageeditor
"""
import os
from glob import glob
from PIL import Image, ImageDraw

import numpy as np
import gradio as gr

from ..src import all_models, default_model
from ..src import det_models, default_det
from ..src.infer_prompt import inference_prompt as infer


def wrap_inference(img, prompt, sam_model, det_model, thresh):
    anns = infer(img, prompt, sam_model, det_model, thresh)
    return (img, anns)


# Define UI settings & layout

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)

    folder_examples = os.path.join(os.path.dirname(__file__), 'assets')
    image_examples = glob(os.path.join(folder_examples, "*.jpg"))
    video_examples = glob(os.path.join(folder_examples, "*.mp4"))
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 📦 Instance Segmentation using Prompt")
        
        with gr.Row():
            original_img = gr.Image(label="Original Image", interactive=True)
            annotated_img = gr.AnnotatedImage(label="Segmented Instances")
            
        with gr.Row():
            prompt = gr.Textbox(label='Object Prompt', placeholder='Insert comma-separated object')

        with gr.Row():
            with gr.Column(scale=7):
                with gr.Accordion(label='Advanced Settings', open=False):
                    with gr.Row():
                        model = gr.Dropdown(all_models, value=default_model, label="Select Model")
                        modet = gr.Dropdown(det_models, value=default_det, label="Select Detector")
                        thresh = gr.Slider(value=0.01, minimum=0, maximum=1.0, step=0.01, label="Detection Threshold",
                                            info="A filtering threshold in [0,1], using the model's detection quality.")
            with gr.Column(scale=1):
                button = gr.Button("Detect & Segment")

        # Events
        button.click(
                fn=wrap_inference, 
            inputs=[original_img, prompt, model, modet, thresh],
            outputs=[annotated_img]
        )
    
    return gui, annotated_img


if __name__ == "__main__":
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000)
