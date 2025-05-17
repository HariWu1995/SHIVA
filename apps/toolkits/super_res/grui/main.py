import gradio as gr
import numpy as np

from PIL import Image
from ..src import (
    load_model_full, run_upsampling_full,
    load_model_face, run_upsampling_face,
)


def inference(image, scale, enhance_face: bool):
    scale = int(scale)
    if scale < 4:
        model_name = "realesgan_x2+"
    else:
        model_name = "realesgan_x4+"
    full_model = load_model_full(model_name)

    if not enhance_face:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        output = run_upsampling_full(full_model, image, outscale=scale)
        return output
    
    model_name = "restoreformer"
    face_model = load_model_face(model_name, upscale=scale, upsampler=full_model)
    
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    output = run_upsampling_face(face_model, image)
    output = Image.fromarray(output)
    return output


# Define UI settings & layout

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 📶 Upscaling")

        with gr.Row():

            with gr.Column():
                img_in = gr.Image(label='Input')
                scaler = gr.Slider(value=2, minimum=1, maximum=100, step=1, label='Scale')
                facenabler = gr.Checkbox(value=False, label="Enhance Face")

            with gr.Column():
                img_out = gr.Image(label='Output')
                button = gr.Button(value="⬆️ Up", variant='primary')
    
        button.click(fn=inference, inputs=[img_in, scaler, facenabler], outputs=img_out)

    return gui, [img_out]


if __name__ == "__main__":
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000, share=False)

