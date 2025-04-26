import gradio as gr

from ..src import *
from .utils import *


# Define UI settings & layout

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## Template")

        with gr.Row():
            img_in = gr.Image(label='Input')
            img_out = gr.Image(label='Output')

        with gr.Row():

            with gr.Column(scale=2, **column_kwargs):
                with gr.Row():
                    run_button = gr.Button(value="Run", variant='primary')
    
        run_button.click(fn=lambda x: x, inputs=img_in, outputs=img_out)

    return gui, [img_out]


if __name__ == "__main__":
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000, share=True)

