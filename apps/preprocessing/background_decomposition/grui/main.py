"""
Emojis: 🎑🌁🌆🖼️🏞️🌄🏜️🏙🗺️🖼🌉🏕️
"""
import gradio as gr

from ..src import all_models, default_model, run_rembg
from .utils import invert_mask, apply_mask, save_all, expand_img


# Define UI settings & layout

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 🖼 Background Decomposition")

        with gr.Row():
            img_in = gr.Image(label='Input')
            img_out = gr.Image(label='Object')
            img_mask = gr.Image(label='Mask')

        with gr.Row():

            with gr.Column(scale=2, **column_kwargs):
                with gr.Row():
                    run_button = gr.Button(value="Decompose", variant='primary')
                    inv_button = gr.Button(value="Invert Mask", variant='secondary')
                with gr.Row():
                    exp_button = gr.Button(value="Expand", variant='secondary')
                    save_button = gr.Button(value="Save", variant='secondary')
                # app_button = gr.Button(value="Apply Mask", variant='secondary')
                # send_button = gr.Button(value="Send ⇩", variant='secondary')

            with gr.Column(scale=4, **column_kwargs):

                model = gr.Dropdown(label="Background Remover", value=default_model, choices=all_models)
                alpha = gr.Checkbox(label="Alpha matting", value=False)
                # mask = gr.Checkbox(label="Return mask", value=False)
                xpand = gr.Checkbox(label="Expansion", value=False)

            with gr.Column(scale=4, visible=False, **column_kwargs) as alpha_mask_options:
                alpha_erosion_size = gr.Slider(label="Erosion size"        , minimum=0, maximum= 40, step=1, value= 10)
                alpha_fg_threshold = gr.Slider(label="Foreground threshold", minimum=0, maximum=255, step=1, value=240)
                alpha_bg_threshold = gr.Slider(label="Background threshold", minimum=0, maximum=255, step=1, value= 10)
                
            with gr.Column(scale=1, visible=False, **column_kwargs) as expansion_options:
                l_size = gr.Slider(label="Left"  , minimum=0, maximum=255, step=1, value=0)
                r_size = gr.Slider(label="Right" , minimum=0, maximum=255, step=1, value=0)
                t_size = gr.Slider(label="Top"   , minimum=0, maximum=255, step=1, value=0)
                b_size = gr.Slider(label="Bottom", minimum=0, maximum=255, step=1, value=0)

        ## Functionality
        rembg_inputs = [img_in, alpha, alpha_fg_threshold, alpha_bg_threshold, alpha_erosion_size, model]
        display_block = lambda x: gr.update(visible=x)

        alpha.change(fn=display_block, inputs=[alpha], outputs=[alpha_mask_options])
        xpand.change(fn=display_block, inputs=[xpand], outputs=[expansion_options])

        # app_button.click(fn=apply_mask, inputs=[img_out, img_mask], outputs=[img_mask])
        exp_button.click(fn=expand_img, inputs=[img_in, 
                                                l_size, r_size, 
                                                t_size, b_size], outputs=[img_in])

        inv_button.click(fn=invert_mask, inputs=[img_mask], outputs=[img_mask])
        run_button.click(fn=run_rembg, inputs=rembg_inputs, outputs=[img_out, img_mask])
        save_button.click(fn=save_all, inputs=[img_out, img_mask], outputs=None)

    return gui, [img_out, img_mask]


if __name__ == "__main__":
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000, share=True)

