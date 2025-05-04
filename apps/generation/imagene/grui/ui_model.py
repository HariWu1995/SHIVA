"""
Emojis: 🗣️🗪🗫🗯💭
"""
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

from glob import glob
from os import path as osp

import gc
import gradio as gr
import numpy as np

from . import shared as ui
from .utils import symbols, gradget

from ..src import shared
from ..src.path import IMAGENE_LOCAL_MODELS, LOCAL_IP_ADAPTERS
from ..src.utils import clear_torch_cache
from ..src.logging import logger
from ..src.pipelines import (
    load_pipeline_generate,
    load_pipeline_variate,
    load_pipeline_inpaint,
    enable_lowvram_usage,
)


def _load_generator(model_selected):
    if  (shared.pipe_name == "Generation") \
    and (shared.model_name == model_selected) \
    and (shared.model is not None):
        return shared.pipe_name, True

    model_version, model_name = model_selected.split('/')
    shared.pipe_name = "Generation"
    shared.adapt_name = None
    shared.brush_name = None
    shared.model_name = model_selected
    shared.model = None
    shared.model = load_pipeline_generate(model_name, model_version)
    return shared.pipe_name, True


def _load_variator(model_selected, adapt_name):
    if  (shared.pipe_name == "Variation") \
    and (shared.adapt_name == adapt_name) \
    and (shared.model_name == model_selected) \
    and (shared.model is not None):
        return shared.pipe_name, True

    model_version, model_name = model_selected.split('/')
    if adapt_name is not None:
        adapt_type = adapt_name.split('/')[2]
    else:
        adapt_type = 'ip_adapter'

    shared.pipe_name = "Variation"
    shared.brush_name = None
    shared.adapt_name = adapt_name
    shared.model_name = model_selected
    shared.model = None
    shared.model = load_pipeline_variate(model_name, model_version, adapt_type)
    return shared.pipe_name, True


def _load_inpaintor(model_selected):
    
    model_version, model_name = model_selected.split('/')

    if  (shared.pipe_name == "Inpainting") \
    and (shared.model_name == model_selected) \
    and (shared.model is not None):
        return shared.pipe_name, True

    shared.pipe_name = "Inpainting"
    shared.adapt_name = None
    shared.brush_name = None
    shared.model_name = model_selected
    shared.model = None
    shared.model = load_pipeline_inpaint(model_name, model_version, None)
    return shared.pipe_name, True


def _load_inbrusher(model_selected, brush_name):
    
    model_version, model_name = model_selected.split('/')
    if model_version == 'sdxl' \
    and brush_name is not None:
        brush_name = brush_name + '_xl'

    if  (shared.pipe_name == "Inbrushing") \
    and (shared.brush_name == brush_name) \
    and (shared.model_name == model_selected) \
    and (shared.model is not None):
        return shared.pipe_name, True

    shared.pipe_name = "Inbrushing"
    shared.adapt_name = None
    shared.brush_name = brush_name
    shared.model_name = model_selected
    shared.model = None
    shared.model = load_pipeline_inpaint(model_name, model_version, brush_name)
    return shared.pipe_name, True


def _load_model(pipeline, model_gen, model_fill, model_brush, model_adapt):
    if pipeline == "Inpainting":
        return _load_inpaintor(model_fill)
    elif pipeline == "Inbrushing":
        return _load_inbrusher(model_gen, model_brush)
    elif pipeline == "Variation":
        return _load_variator(model_gen, model_adapt)
    return _load_generator(model_gen)


def _offload_model(pipeline):
    shared.model = enable_lowvram_usage(shared.model, offload_only=pipeline=="Variation")
    print('\n Model is offloaded')
    logger.info('Model is offloaded')


def _reload_model():
    if shared.pipe_name == "Inpainting":
        return _load_inpaintor(shared.model_name)
    elif shared.pipe_name == "Inbrushing":
        return _load_inbrusher(shared.model_name, shared.brush_name)
    elif shared.pipe_name == "Variation":
        return _load_variator(shared.model_name, shared.adapt_name)
    return _load_generator(shared.model_name)
    

def _unload_model():
    shared.model = None
    gc.collect()
    clear_torch_cache()
    return shared.pipe_name, False


#############################################
#           UI settings & layout            #
#############################################

def create_ui(min_width: int = 25):

    pipelines = ["Generation", "Variation", "Inpainting", "Inbrushing"]
    models_brush = ["random_mask", "segmentation_mask"]
    models_inpaint = []
    models_generate = []

    for m in IMAGENE_LOCAL_MODELS.keys():
        if m.startswith('brush'):
            continue
        elif m.endswith('_inpaint'):
            models_inpaint.append(m)
        else:
            models_generate.append(m)

    ip_adapters = list(LOCAL_IP_ADAPTERS.keys())

    # Layout
    with gr.Accordion(label="Model Loader", open=True) as gui:
        with gr.Row(variant="panel"):

            with gr.Column(min_width=50):
                ui.gradio["pipeline"] = gr.Dropdown(label="Pipeline", value=None, choices=pipelines, interactive=True, visible=True)
                
                gr.Markdown("ℹ️ **Offload**: Send model parts to CPU and call to GPU only when needed. Recommended for low VRAM.")
                gr.Markdown("⚠️ **Warning**: LoRA is not supported with `Offload`")

            with gr.Column(min_width=50):
                ui.gradio["model_gen"] = gr.Dropdown(label="Model", value=None, choices=models_generate, interactive=True, visible=False)
                ui.gradio["model_fill"] = gr.Dropdown(label="Model", value=None, choices=models_inpaint, interactive=True, visible=False)
                with gr.Row(variant="panel"):
                    ui.gradio["model_status"] = gr.Checkbox(label="Model is loaded", interactive=False)
                with gr.Row():
                    with gr.Column(min_width=3):
                        ui.gradio["model_load_refresh"] = gr.Button(value=symbols["refresh"], variant="secondary")
                    with gr.Column(min_width=7):
                        ui.gradio["model_load_trigger"] = gr.Button(value=symbols["trigger"], variant="primary")
                with gr.Row():
                    with gr.Column(min_width=3):
                        ui.gradio["model_load_release"] = gr.Button(value=symbols["release"], variant="secondary", size="md")
                    with gr.Column(min_width=3):
                        ui.gradio["model_load_offload"] = gr.Button(value=symbols["offload"], variant="primary", size="md")

            with gr.Column(min_width=50):
                ui.gradio["model_brush"] = gr.Dropdown(label="BrushNet", choices=models_brush, interactive=True, visible=False, value=None)
                ui.gradio["model_adapt"] = gr.Dropdown(label="IP-Adapter", choices=ip_adapters, interactive=True, visible=False, value=None)

        # Control visibility based on the selected pipeline
        def update_visibility(pipeline):
            return (
                gr.update(visible = (pipeline != "Inpainting")),
                gr.update(visible = (pipeline == "Inpainting")),
                gr.update(visible = (pipeline == "Inbrushing")),
                gr.update(visible = (pipeline ==  "Variation")),
            )

        ui.gradio["pipeline"].change(
                fn=update_visibility, 
            inputs=gradget("pipeline"), 
            outputs=gradget("model_gen", "model_fill", "model_brush", "model_adapt"),
        )

        ## Event handlers
        m_inputs = gradget("pipeline", "model_gen", "model_fill", "model_brush", "model_adapt")
        m_output = gradget("pipeline", "model_status")

        ui.gradio["model_load_trigger"].click(fn=_load_model, inputs=m_inputs, outputs=m_output)
        ui.gradio["model_load_release"].click(fn=_unload_model, inputs=None, outputs=m_output)
        ui.gradio["model_load_refresh"].click(fn=_reload_model, inputs=None, outputs=m_output)
        ui.gradio["model_load_offload"].click(fn=_offload_model, inputs=gradget("pipeline"))

        # Schedule
        timer = gr.Timer(value=1.0)
        timer.tick(fn=lambda: shared.model is not None, outputs=gradget("model_status"))
        
    return gui


if __name__ == "__main__":
    gui = create_ui()
    gui.launch(
          server_name = ui.host, 
          server_port = ui.port, 
                share = ui.share,
            inbrowser = ui.auto_launch,
    )

