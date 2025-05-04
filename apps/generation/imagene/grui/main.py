import gradio as gr

from . import shared as ui
from .utils import gradget
from .ui_loader import create_ui as create_ui_loader
from .ui_wspace import create_ui as create_ui_wspace


# Define UI settings & layout

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 🕋 Imagene")

        with gr.Tab(label='Model Settings'):
            create_ui_loader()

        with gr.Tab(label='Workspace'):
            create_ui_wspace()

        # Control workspace display
        def update_visibility(pipeline):
            return (
                gr.update(visible = (pipeline == "Generation")),
                gr.update(visible = (pipeline ==  "Variation")),
                gr.update(visible = (pipeline in ["Inpainting", "Inbrushing"])),
            )

        ui.gradio["pipeline"].change(
                fn=update_visibility, 
            inputs=gradget("pipeline"), 
            outputs=gradget("tab_gen", "tab_var", "tab_fill"),
        )

    return gui


if __name__ == "__main__":
    gui = create_ui()
    gui.launch(
          server_name = ui.host, 
          server_port = ui.port, 
                share = ui.share,
            inbrowser = ui.auto_launch,
    )

