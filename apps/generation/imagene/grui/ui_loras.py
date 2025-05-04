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
from ..src.path import IMAGENE_LOCAL_LORAS
from ..src.utils import clear_torch_cache
from ..src.logging import logger
from ..src.pipelines import enable_lowvram_usage


def _load_loras(lora_config):
    shared.model.enable_lora()

    for config in lora_config:
        lora_id = config["id"]
        lora_name = config["name"]
        lora_weight = float(config["Weight"])

        if lora_id in shared.lora_names:
            shared.lora_weights[shared.lora_names.index(lora_id)] = lora_weight
            continue

        if lora_weight <= 0:
            logger.warn(f"LoRA {lora_name} has non-positive weight = {lora_weight}")

        lora_path = IMAGENE_LOCAL_LORAS[
                    IMAGENE_LOCAL_LORAS['name'] == lora_name]['path'].values.tolist()[0]
        lora_dir = osp.dirname(lora_path)
        lora_fn = osp.basename(lora_path)
        print(f'\n Loaded LoRA = {lora_fn}')
        logger.info(f'Loaded LoRA = {lora_fn}')

        shared.lora_names.append(lora_id)
        shared.lora_weights.append(lora_weight)

        shared.model.load_lora_weights(lora_dir, adapter_name=lora_id, weight_name=lora_fn)
    shared.model.set_adapters(shared.lora_names, adapter_weights=shared.lora_weights)
    
    shared.model = shared.model.to(device=shared.device)
    # shared.model = enable_lowvram_usage(shared.model)
    return True


def _unload_loras():
    shared.model.disable_lora()
    shared.model.delete_adapters(shared.lora_names)
    shared.lora_names = []
    shared.lora_weights = []
    return False


#############################################
#           UI settings & layout            #
#############################################

def create_ui(min_width: int = 25):

    lora_colors = {'Select': '#ffcccb', 'Weight': 'lightblue'}
    lora_headers = ['sd_version','id','category','article','trigger_words','weight','Select','Weight']
    lora_colwidth = ['5%', '10%', '10%', '20%', '25%', '10%', '10%', '10%']

    lora_longlist = IMAGENE_LOCAL_LORAS[lora_headers[:-2]].copy()
    lora_longlist['Select'] = [False] * len(lora_longlist)
    lora_longlist['Weight'] = [0.0] * len(lora_longlist)

    # Layout
    with gr.Accordion(label="LoRA Loader", open=False) as gui:
        with gr.Row():
            with gr.Column(scale=3):
                ui.gradio["lora_names"] = gr.Dataframe(max_height=250, label="LoRA Info", value=None, interactive=True, headers=lora_headers, show_search="filter", 
                                                                                                                static_columns=lora_headers[:-2], 
                                                                                                                column_widths=lora_colwidth)
            with gr.Column(scale=1):
                ui.gradio["lora_select"] = gr.JSON(max_height=250, label="LoRA Select", value=None, container=True)
                with gr.Row(variant="panel"):
                    ui.gradio["lora_status"] = gr.Checkbox(label="LoRA(s) are loaded", interactive=False)
                with gr.Row():
                    with gr.Column(min_width=5):
                        ui.gradio["loras_trigger"] = gr.Button(value=symbols["trigger"], variant="primary")
                    with gr.Column(min_width=3):
                        ui.gradio["loras_release"] = gr.Button(value=symbols["release"], variant="secondary")

        def update_loras(model_name):
            model_ver = model_name.split('/')[0]
            model_loras = lora_longlist[lora_longlist['sd_version'] == model_ver].copy()
            return model_loras

        if "model_gen" in list(ui.gradio.keys()):
            ui.gradio["model_gen"].change(
                    fn=update_loras, 
                inputs=gradget("model_gen"), 
                outputs=gradget("lora_names"), 
            )

        # NOTE: Not work in `interactive` mode
        # Stylization: 
        #   https://www.gradio.app/guides/styling-the-gradio-dataframe
        #   https://stackoverflow.com/questions/75211791/pandas-dataframe-style-not-blank-different-columns

        def stylize_loras(model_loras):
            highlight_col = lambda s: s.mask(s.notna(), f'background-color: {lora_colors.get(s.name, "none")}')
            return model_loras.style.apply(highlight_col, axis=0)

        def select_loras(model_loras):
            loras = model_loras[model_loras['Select'].isin(['true', 'True', True, 1])]
            loras = loras[['id','Weight']].merge(IMAGENE_LOCAL_LORAS[['id','name']], how='left', on=['id'])
            return loras.to_dict(orient='records')

        ui.gradio["lora_names"].change(
                fn=stylize_loras,
            inputs=gradget("lora_names"), 
            outputs=gradget("lora_names"), 
        ).then(
                fn=select_loras,
            inputs=gradget("lora_names"), 
            outputs=gradget("lora_select"), 
        )

        ## Event handlers
        l_inputs = gradget("lora_select")
        l_output = gradget("lora_status")

        ui.gradio["loras_trigger"].click(fn=_load_loras, inputs=l_inputs, outputs=l_output)
        ui.gradio["loras_release"].click(fn=_unload_loras, inputs=None, outputs=l_output)

        # Schedule
        timer = gr.Timer(value=1.0)
        timer.tick(fn=lambda: len(shared.lora_names), outputs=gradget("lora_status"))

    return gui


if __name__ == "__main__":
    gui = create_ui()
    gui.launch(
          server_name = ui.host, 
          server_port = ui.port, 
                share = ui.share,
            inbrowser = ui.auto_launch,
    )

