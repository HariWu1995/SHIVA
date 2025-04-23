"""
Emojis: 🗣️🗪🗫🗯💭
Reference: 
    https://www.gradio.app/docs/gradio/chatinterface
    https://www.gradio.app/docs/gradio/multimodaltextbox
    https://www.gradio.app/guides/multimodal-chatbot-part1
    https://www.gradio.app/guides/creating-a-chatbot-fast
"""
import yaml
import uuid
import datetime as dt
import gradio as gr

from . import shared as ui
from .utils import gradget, symbols
from .inference import run_inference
# from .inference_sample import run_inference

from ..src import shared
from ..src.logging import logger


session_id = str(uuid.uuid4())


def feedback(x: gr.LikeData):
    logger.info(f"[Dialog][Feedback] Index = {x.index} - Value = {x.value} - Like = {x.liked}")


def add_request(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False, **ui.chatbox_settings)


def remove_request(history):
    if len(history) > 0:
        _ = history.pop(-1)
    return history


def clear_history(history):
    return []


def save_history(history):
    save_timestamp = int(dt.datetime.now().timestamp())
    save_filepath = shared.model_args.disk_cache_dir / f"{session_id}-{save_timestamp}.yaml"
    with open(save_filepath, 'w') as file:
        yaml.dump(history, file)
    return history


def gather_states(*args):
    gen_states = dict()
    for k, v in zip(ui.generation_params, args):
        gen_states[k] = v
    return gen_states


#############################################
#           UI settings & layout            #
#############################################

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        # ui.gradio['chat_request'] = gr.State()
        # ui.gradio['chat_history'] = gr.JSON(visible=False)

        with gr.Tab("Chatbox"):
            ui.gradio["chatbot"] = gr.Chatbot(elem_id="chatbot", type="messages", bubble_full_width=False)
            with gr.Row():
                with gr.Column(scale=10):
                    ui.gradio["chatbox"] = gr.MultimodalTextbox(interactive=True, **ui.chatbox_settings)
                with gr.Column(scale=1, min_width=2):
                    with gr.Row():
                        ui.gradio["chat_reset"] = gr.Button(symbols['reset'])
                    with gr.Row():
                        ui.gradio["chat_undo" ] = gr.Button(symbols['undo'])
                    with gr.Row():
                        ui.gradio["chat_save" ] = gr.Button(symbols['save'])

        with gr.Tab('Control Panel', elem_id='chat-controls'):
            ui.gradio['start_with'] = gr.Textbox(value=ui.settings['start_with'], label='Reply with', placeholder='Yes please!')
            ui.gradio['user_bio'] = gr.Textbox(value=ui.settings['user_bio'], label='Start reply with', placeholder='Yes please!')

            ui.gradio['chat_mode'] = gr.Radio(value=ui.settings['chat_mode'] 
                                                if ui.settings['chat_mode'] in ['chat', 'chat-instruct'] else None, 
                                            choices=['chat', 'chat-instruct', 'instruct'], 
                                            label='Chat Mode', elem_id='chat-mode', 
                                            info='Impact how to generate prompt. In `instruct` and `chat-instruct` modes, instruction template is used.')

            ui.gradio['chat-instruct_cmd'] = gr.Textbox(value=ui.settings['chat-instruct_cmd'], lines=12, 
                                                        label='Command for chat-instruct mode', 
                                                         info='<|character|> and <|prompt|> get replaced with `bot name` and `chat prompt` respectively.', 
                                                        visible=ui.settings['chat_mode'] == 'chat-instruct')
        # Interface state
        gen_params_state = ui.generation_params
        ui.gradio['state'] = gr.State({k: None for k in gen_params_state})

        # Functionalities
        chat_block = lambda: gr.MultimodalTextbox(value=None, interactive=False, **ui.chatbox_settings)
        chat_unblock = lambda: gr.MultimodalTextbox(value=None, interactive=True, **ui.chatbox_settings)

        ## Event handlers
        ui.gradio["chatbot"].like(feedback, None, None, like_user_message=True)
        ui.gradio["chatbox"].submit(add_request, [ui.gradio["chatbot"], ui.gradio["chatbox"]], 
                                                 [ui.gradio["chatbot"], ui.gradio["chatbox"]])\
                            .then(chat_block, None, ui.gradio["chatbox"])\
                            .then(gather_states, gradget(gen_params_state), gradget('state'))\
                            .then(run_inference, gradget("state","chatbot"), gradget('chatbot'))\
                            .then(chat_unblock, None, ui.gradio["chatbox"])\
                            .then(None, None, None, js=ui.js["audio_noti"])

        # Meta handlers
        chat_history = ui.gradio["chatbot"]

        ui.gradio["chat_undo"].click(remove_request, chat_history, chat_history)
        ui.gradio["chat_reset"].click(clear_history, chat_history, chat_history)
        ui.gradio["chat_save"].click(save_history, chat_history, chat_history)

    return gui


if __name__ == "__main__":
    gui = create_ui()
    gui.launch(
          server_name = ui.host, 
          server_port = ui.port, 
                share = ui.share,
            inbrowser = ui.auto_launch,
        allowed_paths = ui.allowed_paths,
    )

