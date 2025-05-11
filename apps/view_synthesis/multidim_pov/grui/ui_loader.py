import itertools as it
import gradio as gr

from . import shared as ui
from .utils import gradget

from ..src import shared
from ..src.utils import clear_torch_cache
from ..src.models import load_diff360_i2p, load_diff360_t2p, load_mvdiff_t2p, load_mvdiff_i2p


def wrap_loader(model_selected: str):
    pipeline, model_name = model_selected.split('/')

    if  (shared.model_base is not None) \
    and (shared.model_name == model_name) \
    and (shared.pipeline == pipeline):
        return model_selected, True

    model = None

    if pipeline == 'txt2pano':
        if model_name == 'mvdiffusion':
            model = load_mvdiff_t2p()
        elif model_name == 'diffusion360':
            model = load_diff360_t2p()

    elif pipeline == 'img2pano':
        if model_name == 'mvdiffusion':
            model = load_mvdiff_i2p()
        elif model_name == 'diffusion360':
            model = load_diff360_i2p()

    if model is None:
        raise ValueError(f'({pipeline} - {model_name}) is not supported!')

    shared.model_base = model
    shared.model_name = model_name
    shared.pipeline = pipeline
    return model_selected, True


def wrap_unloader():
    shared.model = None
    clear_torch_cache()
    return f"{shared.pipeline}/{shared.model_name}", False


def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    models = ['mvdiffusion', 'diffusion360']
    pipelines = ['txt2pano', 'img2pano']
    base_models = [f'{p}/{m}' for p, m in it.product(pipelines, models)]

    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        with gr.Row(variant='panel'):

            with gr.Column():
                gr.Markdown("#### ⭐ Base Model")

                ui.gradio['model_base'] = gr.Dropdown(choices=base_models, label='Model', value=None)
                ui.gradio['status_base'] = gr.Checkbox(interactive=False, label="Model is loaded")

                with gr.Row():
                    with gr.Column(scale=1, min_width=5):
                        gr.Markdown('')
                    with gr.Column(scale=3, min_width=25):
                        ui.gradio['model_base_loader'] = gr.Button(value=ui.symbols["trigger"], variant="primary")
                    with gr.Column(scale=3, min_width=25):
                        ui.gradio['model_base_deleter'] = gr.Button(value=ui.symbols["delete"], variant="secondary")
                    with gr.Column(scale=1, min_width=5):
                        gr.Markdown('')

            with gr.Column():
                gr.Markdown("#### ✨ Refine Model")

                ui.gradio['model_refine'] = gr.Dropdown(choices=base_models, label='Model', value=None)
                ui.gradio['status_refine'] = gr.Checkbox(interactive=False, label="Model is loaded")

                with gr.Row():
                    with gr.Column(scale=1, min_width=5):
                        gr.Markdown('')
                    with gr.Column(scale=3, min_width=25):
                        ui.gradio['model_refine_loader'] = gr.Button(value=ui.symbols["trigger"], variant="primary")
                    with gr.Column(scale=3, min_width=25):
                        ui.gradio['model_refine_deleter'] = gr.Button(value=ui.symbols["delete"], variant="secondary")
                    with gr.Column(scale=1, min_width=5):
                        gr.Markdown('')
        
        # Event Handle
        inputs = gradget(['model_base'])
        outputs = gradget(['model_base','status_base'])

        ui.gradio['model_base_loader'].click(fn=wrap_loader, inputs=inputs, outputs=outputs)
        ui.gradio['model_base_deleter'].click(fn=wrap_unloader, inputs=None, outputs=outputs)

        # Schedule
        timer = gr.Timer(value=1.0)
        timer.tick(fn=lambda: shared.model is not None, inputs=None, outputs=ui.gradio['status'])

    return gui


if __name__ == "__main__":
    gui= create_ui()
    gui.launch(server_name='localhost', server_port=8000)

