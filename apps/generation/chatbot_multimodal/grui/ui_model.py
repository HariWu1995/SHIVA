import re
import math
import psutil
import importlib
import traceback
from functools import partial
from pathlib import Path

import torch
from transformers import is_torch_npu_available, is_torch_xpu_available

import gradio as gr

from . import shared as ui
from .logging import logger
from .utils import (
    gradget, 
    create_refresh_button,
    update_model_parameters, 
    list_interface_input_elements, 
    apply_interface_values, 
    gather_interface_values, 
    apply_model_settings_to_state, 
    save_model_settings, 
    save_instruction_template,
)

from ..src import shared
from ..src.path import MLM_LOCAL_MODELS, LORA_LOCAL_MODELS
from ..src.sampler import loaders_and_params, get_all_params
from ..src.loaders import get_model_metadata, load_model, unload_model, ModelDownloader
from ..src.utils import get_all_device_memory, get_available_instructions


def get_gpu_memory_keys():
    return [k for k in ui.gradio if k.startswith('gpu_memory')]


def get_all_params_wrapper():
    all_params = get_all_params()
    # if 'gpu_memory' in all_params:
    #     all_params.remove('gpu_memory')
    #     for k in get_gpu_memory_keys():
    #         all_params.add(k)
    return sorted(all_params)


def make_loader_params_visible(loader):
    params = []
    all_params = get_all_params_wrapper()
    if loader in loaders_and_params:
        params = loaders_and_params[loader]

        if 'gpu_memory' in params:
            params.remove('gpu_memory')
            params += get_gpu_memory_keys()

    return [gr.update(visible=True) if k in params else 
            gr.update(visible=False) for k in all_params]


def get_default_gpu_mem():
    default_gpu_mem = []
    if shared.args.gpu_memory is not None and \
    len(shared.args.gpu_memory) > 0:
        for m in shared.args.gpu_memory:
            if isinstance(m, int):
                default_gpu_mem.append(m)
            elif 'mib' in m.lower():
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', m)))
            else:
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', m)) * 1000)
    return default_gpu_mem


def get_default_cpu_mem():
    total_cpu_mem = math.floor(psutil.virtual_memory().total / (1024 * 1024))
    m = shared.args.cpu_memory
    if m is None:
        return 0
    if isinstance(m, int):
        return m
    elif isinstance(m, str):
        return re.sub('[a-zA-Z ]', '', m)


def create_ui():

    is_single_user = not ui.multi_user

    # Finding the default values for the GPU and CPU memories
    total_gpu_mem = get_all_device_memory()
    default_gpu_mem = get_default_gpu_mem()
    default_cpu_mem = get_default_cpu_mem()

    # Get available options
    all_models = list(MLM_LOCAL_MODELS.keys())
    all_loras = list(LORA_LOCAL_MODELS.keys())
    all_loaders = loaders_and_params.keys()
    all_instructions = get_available_instructions()

    while len(default_gpu_mem) < len(total_gpu_mem):
        default_gpu_mem.append(0)

    with gr.Tab("Model", elem_id="model-tab"):

        with gr.Row():
    
            with gr.Column():
                with gr.Row():
                    ui.gradio['model_menu'] = gr.Dropdown(choices=all_models, value=lambda: shared.model_name, 
                                                                    label='Model', elem_classes='slim-dropdown', interactive=is_single_user)
                    create_refresh_button(
                        ui.gradio['model_menu'], lambda: None, 
                                                 lambda: {'choices': all_models}, elem_classes='refresh-button', interactive=is_single_user)

                    ui.gradio[  'load_model'       ] = gr.Button("Load"         , elem_classes='refresh-button', interactive=is_single_user, visible=not ui.settings['autoload_model'])
                    ui.gradio['unload_model'       ] = gr.Button("Unload"       , elem_classes='refresh-button', interactive=is_single_user)
                    ui.gradio['save_model_settings'] = gr.Button("Save settings", elem_classes='refresh-button', interactive=is_single_user)

            with gr.Column():
                with gr.Row():
                    ui.gradio['lora_menu'] = gr.Dropdown(multiselect=True, 
                                                            choices=all_loras, 
                                                            value=shared.lora_names, 
                                                                    label='LoRA(s)', elem_classes='slim-dropdown', interactive=is_single_user)
                    create_refresh_button(
                        ui.gradio['lora_menu'], lambda: None, 
                                                lambda: {'choices': all_loras, 
                                                        'value': shared.lora_names}, elem_classes='refresh-button', interactive=is_single_user)
                    ui.gradio['lora_menu_apply'] = gr.Button(value='Inject LoRAs', elem_classes='refresh-button', interactive=is_single_user)

        with gr.Row():

            with gr.Column():
                ui.gradio['loader'] = gr.Dropdown(label="Model loader", choices=all_loaders, value=None)

            with gr.Column():
                with gr.Row():
                    ui.gradio['autoload_model'] = gr.Checkbox(value=ui.settings['autoload_model'], 
                                                              label='Autoload the model', 
                                                               info=ui.element_description['model_autoload'], interactive=is_single_user)

                with gr.Tab("Download"):
                    ui.gradio['custom_model_menu'] = gr.Textbox(label="Download model or LoRA", 
                                                                info=ui.element_description['model_download'], interactive=is_single_user)
                    ui.gradio['download_specific_file'] = gr.Textbox(placeholder="File name (for GGUF models)", 
                                                                      show_label=False, max_lines=1, interactive=is_single_user)
                    with gr.Row():
                        ui.gradio['download_model_button'] = gr.Button("Download", variant='primary', interactive=is_single_user)
                        ui.gradio['get_file_list'] = gr.Button("Get file list", interactive=is_single_user)

                with gr.Tab("llamacpp_HF creator"):
                    with gr.Row():
                        ui.gradio['gguf_menu'] = gr.Dropdown(choices=['None'], value=lambda: shared.model_name, label='Choose your GGUF', elem_classes='slim-dropdown', interactive=is_single_user)
                        create_refresh_button(ui.gradio['gguf_menu'], lambda: None, lambda: {'choices': ["None"]}, 'refresh-button', interactive=is_single_user)

                    ui.gradio['unquantized_url'] = gr.Textbox(label="Enter the URL for the original (unquantized) model", info="Example: https://huggingface.co/lmsys/vicuna-13b-v1.5", max_lines=1)
                    ui.gradio['create_llamacpp_hf_button'] = gr.Button("Submit", variant="primary", interactive=is_single_user)
                    gr.Markdown(ui.element_description["llamacpp_hf_creation"])

                with gr.Tab("Customization"):

                    with gr.Row():
                        ui.gradio['customized_template'] = gr.Dropdown(
                            choices=all_instructions, 
                            value='None', 
                            label='Select the desired instruction template', 
                            elem_classes='slim-dropdown',
                        )
                        create_refresh_button(
                            ui.gradio['customized_template'], 
                            lambda: None, 
                            lambda: {'choices': all_instructions}, 
                            elem_classes='refresh-button', 
                            interactive=is_single_user,
                        )

                    ui.gradio['customized_template_submit'] = gr.Button("Submit", variant="primary", interactive=is_single_user)
                    gr.Markdown(ui.element_description["model_customization"])

                with gr.Row():
                    ui.gradio['model_status'] = gr.Markdown('No model is loaded' if shared.model_name == 'None' else 'Ready')


def create_event_handlers():
    ui.gradio['loader'].change(
        make_loader_params_visible, 
        gradget('loader'), 
        gradget(get_all_params_wrapper()), 
        show_progress=False
    )

    # In this event handler, the interface state is read and updated
    # with the model defaults (if any), and then the model is loaded
    # unless "autoload_model" is unchecked
    ui.gradio['model_menu'].change(
        gather_interface_values, 
        gradget(ui.input_elements), 
        gradget('interface_state')
    ).then(
        handle_load_model_event_initial, 
        gradget('model_menu', 'interface_state'), 
        gradget(list_interface_input_elements()) + gradget('interface_state'), 
        show_progress=False
    ).then(
        load_model_wrapper, 
        gradget('model_menu', 'loader', 'autoload_model'), 
        gradget('model_status'), 
        show_progress=True
    ).success(
        handle_load_model_event_final, 
        gradget('truncation_length', 'loader', 'interface_state'), 
        gradget('truncation_length', 'filter_by_loader'), 
        show_progress=False
    )

    ui.gradio['load_model'].click(
        gather_interface_values, 
        gradget(ui.input_elements), 
        gradget('interface_state')
    ).then(
        update_model_parameters, 
        gradget('interface_state'), 
        None
    ).then(
        partial(load_model_wrapper, autoload=True), 
        gradget('model_menu', 'loader'), 
        gradget('model_status'), 
        show_progress=True
    ).success(
        handle_load_model_event_final, 
        gradget('truncation_length', 'loader', 'interface_state'), 
        gradget('truncation_length', 'filter_by_loader'), 
        show_progress=False
    )

    ui.gradio['unload_model'].click(
        handle_unload_model_click, 
        None, 
        gradget('model_status'), 
        show_progress=False
    )

    ui.gradio['save_model_settings'].click(
        gather_interface_values, 
        gradget(ui.input_elements), 
        gradget('interface_state')
    ).then(
        save_model_settings, 
        gradget('model_menu', 'interface_state'), 
        gradget('model_status'), 
        show_progress=False
    )

    ui.gradio['lora_menu_apply'].click(
        load_lora_wrapper, 
        gradget('lora_menu'), 
        gradget('model_status'), 
        show_progress=False
    )

    ui.gradio['download_model_button'].click(
        download_model_wrapper, 
        gradget('custom_model_menu', 'download_specific_file'), 
        gradget('model_status'), 
        show_progress=True)

    ui.gradio['get_file_list'].click(
        partial(download_model_wrapper, return_links=True), 
        gradget('custom_model_menu', 'download_specific_file'), 
        gradget('model_status'), 
        show_progress=True)

    ui.gradio['autoload_model'].change(
        lambda x: gr.update(visible=not x), 
        gradget('autoload_model'), 
        gradget('load_model')
    )

    ui.gradio['create_llamacpp_hf_button'].click(
        create_llamacpp_hf, 
        gradget('gguf_menu', 'unquantized_url'), 
        gradget('model_status'), 
        show_progress=True)

    ui.gradio['customized_template_submit'].click(
        save_instruction_template, 
        gradget('model_menu', 'customized_template'), 
        gradget('model_status'), 
        show_progress=True)


def load_model_wrapper(selected_model, loader, autoload=False):
    settings = get_model_metadata(selected_model)

    if not autoload:
        yield "### {}\n\n- Settings updated: Click \"Load\" to load the model\n- Max sequence length: {}".format(selected_model, ui.settings['truncation_length_info'])
        return

    if selected_model == 'None':
        yield "No model selected"
    else:
        try:
            yield f"Loading `{selected_model}`..."
            unload_model()
            if selected_model != '':
                shared.model, shared.tokenizer = load_model(selected_model, loader)

            if shared.model is not None:
                yield f"Successfully loaded `{selected_model}`."
            else:
                yield f"Failed to load `{selected_model}`."
        except:
            exc = traceback.format_exc()
            logger.error('Failed to load the model.')
            print(exc)
            yield exc.replace('\n', '\n\n')


def load_lora_wrapper(selected_loras):
    yield ("Applying the following LoRAs to {}:\n\n{}".format(shared.model_name, '\n'.join(selected_loras)))
    from ..src.LoRA import add_lora_to_model
    add_lora_to_model(selected_loras)
    yield ("Successfuly applied the LoRAs")


def download_model_wrapper(repo_id, specific_file, progress=gr.Progress(), return_links=False, check=False):
    try:
        if repo_id == "":
            yield ("Please enter a model path")
            return

        repo_id = repo_id.strip()
        specific_file = specific_file.strip()
        downloader = ModelDownloader()

        progress(0.0)
        model, branch = downloader.sanitize_model_and_branch_names(repo_id, None)

        yield ("Getting the download links from Hugging \Face")
        links, sha256, \
        is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=False, specific_file=specific_file)

        if return_links:
            output = "```\n"
            for link in links:
                output += f"{Path(link).name}" + "\n"

            output += "```"
            yield output
            return

        yield ("Getting the output folder")
        output_folder = downloader.get_output_folder(
            model,
            branch,
            is_lora,
            is_llamacpp=is_llamacpp,
            model_dir=shared.args.model_dir 
                    if shared.args.model_dir != shared.args_default.model_dir else None
        )

        if output_folder == Path("models"):
            output_folder = Path(shared.args.model_dir)
        elif output_folder == Path("loras"):
            output_folder = Path(shared.args.lora_dir)

        if check:
            progress(0.5)

            yield ("Checking previously downloaded files")
            downloader.check_model_files(model, branch, links, sha256, output_folder)
            progress(1.0)
        else:
            yield (f"Downloading file{'s' if len(links) > 1 else ''} to `{output_folder}/`")
            downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=is_llamacpp)

            yield (f"Model successfully saved to `{output_folder}/`.")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def create_llamacpp_hf(gguf_name, unquantized_url, progress=gr.Progress()):
    try:
        downloader = ModelDownloader()

        progress(0.0)
        model, branch = downloader.sanitize_model_and_branch_names(unquantized_url, None)

        yield ("Getting the tokenizer files links from Hugging Face")
        links, sha256, \
        is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=True)
        output_folder = Path(shared.args.model_dir) / (re.sub(r'(?i)\.gguf$', '', gguf_name) + "-HF")

        yield (f"Downloading tokenizer to `{output_folder}/`")
        downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=False)

        # Move the GGUF
        (Path(shared.args.model_dir) / gguf_name).rename(output_folder / gguf_name)
        yield (f"Model saved to `{output_folder}/`.\n\nYou can now load it using llamacpp_HF.")

    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def update_truncation_length(current_length, state):
    if 'loader' in state:
        if state['loader'].lower().startswith('exllama'):
            return state['max_seq_len']
        elif state['loader'] in ['llama.cpp', 'llamacpp_HF']:
            return state['n_ctx']

    return current_length


def handle_load_model_event_initial(model, state):
    state = apply_model_settings_to_state(model, state)
    output = apply_interface_values(state)
    _none = update_model_parameters(state)
    return output + [state]


def handle_load_model_event_final(truncation_length, loader, state):
    truncation_length = update_truncation_length(truncation_length, state)
    return [truncation_length, loader]


def handle_unload_model_click():
    unload_model()
    return "Model unloaded"

