import re
import base64
import torch
import gradio as gr

from io import BytesIO
from functools import partial

import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parents[3])
sys.path.append(root_dir)

from src import shared
from grui import shared as ui
from grui.logging import logger
from grui.extensions.multimodal.multimodal_embedder import MultimodalEmbedder


params = {
    "add_all_images_to_prompt": False,
    "vision_device": None,          # device to run vision encoder
    "vision_bits": 32,              # bits to load vision encoder in, either 16 or 32
    "projector_device": None,       # device to run multimodal projector on
    "projector_bits": 32,           # multimodal projector bits, either 16 or 32
}


# If 'state' is True, will hijack the next chat generation
input_hijack = {
    'state': False,
    'value': ["", ""]
}


# initialized in ui, so that params are loaded from settings
multimodal_embedder: MultimodalEmbedder = None


def chat_input_modifier(text, visible_text, state):
    global input_hijack
    if input_hijack['state']:
        input_hijack['state'] = False
        return input_hijack['value'](text, visible_text)
    else:
        return text, visible_text


def add_chat_picture(picture, text, visible_text):
    # resize the image, so that shortest edge is 
    #           at least 224 (size for CLIP), 
    #       and at most 300 (to keep history manageable)
    # Adjusted to 336 for the values here, due to the increased resolution in llava-v1.5
    max_hw = max(picture.size)
    min_hw = min(picture.size)
    aspect_ratio = max_hw / min_hw
    
    shortest_edge = int(max(336 / aspect_ratio, 336))
    longest_edge = int(shortest_edge * aspect_ratio)

    w = shortest_edge if picture.width < picture.height else longest_edge
    h = shortest_edge if picture.width >= picture.height else longest_edge
    picture = picture.resize((w, h))

    buffer = BytesIO()
    picture.save(buffer, format="PNG")

    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    image = f'<img src="data:image/jpeg;base64,{img_str}">'

    if '<image>' in text:
        text = text.replace('<image>', image)
    else:
        text = image + '\n' + text

    if visible_text == '' or visible_text is None:
        visible_text = text
    elif '<image>' in visible_text:
        visible_text = visible_text.replace('<image>', image)
    else:
        visible_text = visible_text + '\n' + image

    return text, visible_text


def custom_tokenized_length(prompt):
    return multimodal_embedder.len_in_tokens(prompt)


def tokenizer_modifier(state, prompt, input_ids, input_embeds):
    global params
    image_match = re.search(r'<img src="data:image/jpeg;base64,[A-Za-z0-9+/=]+">', prompt)

    if image_match is None:
        return prompt, input_ids, input_embeds

    prompt, input_ids, input_embeds, \
                       total_embedded = multimodal_embedder.forward(prompt, state, params)
    return (
            prompt,
           input_ids.unsqueeze(0).to(device=shared.model.device, dtype=torch.int64),
        input_embeds.unsqueeze(0).to(device=shared.model.device, dtype=shared.model.dtype)
    )


def create_ui():
    global multimodal_embedder
    multimodal_embedder = MultimodalEmbedder(params)

    with gr.Column():
        picture_select = gr.Image(label='Send a picture', type='pil')
        uni_image_check = gr.Checkbox(False, label='Embed all images, not only the last one', 
                                              info='Model do not work well with multiple images')
    # Prepare the input hijack
    picture_select_fn = lambda p: input_hijack.update({"state": True, "value": partial(add_chat_picture, p)})
    picture_select.upload(picture_select_fn, [picture_select], None)
    
    picture_select.clear(lambda: input_hijack.update({"state": False, "value": ["", ""]}), None, None)
    uni_image_check.change(lambda x: params.update({"add_all_images_to_prompt": x}), uni_image_check, None)
    
    ui.gradio['Generate'].click(lambda: None, None, picture_select)
    ui.gradio['textbox'].submit(lambda: None, None, picture_select)

