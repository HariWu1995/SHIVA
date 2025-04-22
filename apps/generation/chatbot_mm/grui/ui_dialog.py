"""
Emojis: 🗣️🗪🗫🗯💭
Reference: 
    https://www.gradio.app/docs/gradio/chatinterface
    https://www.gradio.app/docs/gradio/multimodaltextbox
    https://www.gradio.app/guides/multimodal-chatbot-part1
    https://www.gradio.app/guides/creating-a-chatbot-fast
"""
import gradio as gr

from . import shared as ui
from ..src import shared
from .utils import gather_interface_values, list_interface_input_elements, gradget
from .inference import run_inference
# from .inference_sample import run_inference


def feedback(x: gr.LikeData):
    print(f"\n[Dialog - Feedback] Index = {x.index} - Value = {x.value} - Like = {x.liked}")


def add_request(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def gather_states(*args):
    states = gather_interface_values(*args, elements=ui.input_elements)
    return states


#############################################
#           UI settings & layout            #
#############################################

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 🗫 Multi-Modal Dialogue")

        # ui.gradio['chat_request'] = gr.State()
        # ui.gradio['chat_history'] = gr.JSON(visible=False)

        with gr.Tab("Chatbox"):
            ui.gradio["chatbot"] = gr.Chatbot(elem_id="chatbot", type="messages", bubble_full_width=False)
            ui.gradio["chatbox"] = gr.MultimodalTextbox(
                sources=["microphone", "upload"],
                file_count="multiple",
                placeholder="Enter message or upload file...",
                show_label=False,
                interactive=True,
            )

        with gr.Tab('Control Panel', elem_id='chat-controls'):
            ui.gradio['start_with'] = gr.Textbox(value=ui.settings['start_with'], label='Reply with', placeholder='Yes please!')
            ui.gradio['user_bio'] = gr.Textbox(value=ui.settings['user_bio'], label='Start reply with', placeholder='Yes please!')

            ui.gradio['chat_mode'] = gr.Radio(value=ui.settings['chat_mode'] 
                                                if ui.settings['chat_mode'] in ['chat', 'chat-instruct'] else 'None', 
                                            choices=['chat', 'chat-instruct', 'instruct', 'None'], 
                                            label='Chat Mode', elem_id='chat-mode', 
                                            info='Impact how to generate prompt. In `instruct` and `chat-instruct` modes, instruction template is used.')

            ui.gradio['chat-instruct_cmd'] = gr.Textbox(value=ui.settings['chat-instruct_cmd'], lines=12, 
                                                        label='Command for chat-instruct mode', 
                                                         info='<|character|> and <|prompt|> get replaced with `bot name` and `chat prompt` respectively.', 
                                                        visible=ui.settings['chat_mode'] == 'chat-instruct')
        # Interface state
        ui.input_elements = [x for x in list_interface_input_elements() if x in ui.gradio.keys()]
        ui.gradio['state'] = gr.State({k: None for k in ui.input_elements})

        # Functionalities
        chat_block = lambda: gr.MultimodalTextbox(interactive=False)
        chat_unblock = lambda: gr.MultimodalTextbox(interactive=True)

        ## Event handlers
        ui.gradio["chatbot"].like(feedback, None, None, like_user_message=True)
        ui.gradio["chatbox"].submit(add_request, [ui.gradio["chatbot"], ui.gradio["chatbox"]], 
                                                 [ui.gradio["chatbot"], ui.gradio["chatbox"]])\
                            .then(chat_block, None, ui.gradio["chatbox"])\
                            .then(gather_states, gradget(ui.input_elements), gradget('state'))\
                            .then(run_inference, gradget("state","chatbot"), gradget('chatbot'))\
                            .then(chat_unblock, None, ui.gradio["chatbox"])\
                            .then(None, None, None, js=ui.js["audio_noti"])

        # Schedule
        def get_chat_mode(current_mode: str = 'None'):
            if current_mode not in [None, 'None']:
                return current_mode
            for mode in ['instruct','chat']:
                if shared.model_name.endswith(mode):
                    return mode
            return current_mode

        timer = gr.Timer(value=10)
        timer.tick(fn=get_chat_mode, inputs=gradget("chat_mode"), outputs=gradget("chat_mode"))

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

