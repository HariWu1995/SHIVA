import os
from glob import glob
from PIL import Image, ImageDraw

import numpy as np
import gradio as gr

from .utils import run_inference
from .ui_args import load_arguments, update_arguments
from .ui_models import load_annotators


# Define UI settings & layout

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 🕹️ Controlnet Annotation")
        
        with gr.Row():
            img = gr.Image(label="Original Image")
            ann = gr.Image(label="Annotated Image")
        
        with gr.Row():

            with gr.Column(scale=1, variant="panel"):
                model = load_annotators()
                run_btn = gr.Button("Process", variant="primary")

            with gr.Column(scale=7):
                with gr.Accordion("Configuration"):
                    with gr.Row():
                        config_list = load_arguments()

        # Events
        model.change(fn=update_arguments, inputs=[model] + config_list, outputs=config_list)
        run_btn.click(fn=run_inference, inputs=[model, img] + config_list, outputs=[ann])
    
    return gui, ann


if __name__ == "__main__":
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000)

