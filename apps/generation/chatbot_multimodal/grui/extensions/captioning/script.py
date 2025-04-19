from io import BytesIO
import base64

import torch
from transformers import BlipForConditionalGeneration as BlipGenerator, BlipProcessor

import gradio as gr

import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parents[3])
sys.path.append(root_dir)

from grui import wrapper, shared, utils


input_hijack = {
    'state': False,
    'value': ["", ""],
}

processor = None
generator = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def init_model():
    global processor, generator
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    generator = BlipGenerator.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float32)


def chat_input_modifier(text, visible_text, state):
    global input_hijack
    if input_hijack['state']:
        input_hijack['state'] = False
        return input_hijack['value']
    else:
        return text, visible_text


def caption_image(raw_image):

    global generator, processor
    if generator is None:
        init_model()

    generator = generator.to(device=DEVICE)

    inputs = processor(raw_image.convert('RGB'), return_tensors="pt")
    inputs = inputs.to(device=DEVICE, dtype=torch.float32)

    output = generator.generate(**inputs, max_new_tokens=100)
    output = processor.decode(output[0], skip_special_tokens=True)

    generator = generator.to(device="cpu")      # Offload to CPU
    return output


def generate_chat_picture(picture, name1, name2):
    text = f'*{name1} sends {name2} a picture that contains the following: “{caption_image(picture)}”*'
    # lower the resolution of sent images for the chat, 
    # otherwise the log size gets out of control quickly in visible history
    picture.thumbnail((256, 258))
    buffer = BytesIO()
    picture.save(buffer, format="JPEG")

    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    visible_text = f'<img src="data:image/jpeg;base64,{img_str}" alt="{text}">'
    return text, visible_text


def ui(chat_inputs: list | tuple = []):

    picture_select = gr.Image(label='Send a picture', type='pil')
    picture_select_fn = lambda p, n1, n2: \
        input_hijack.update({
            "state": True,
            "value": generate_chat_picture(p, n1, n2)
        })

    # Prepare the input hijack, update the interface values, 
    # call the generation function, and clear the picture
    picture_select.upload(
            fn=picture_select_fn, 
        inputs=[picture_select, shared.gradio['name1'], shared.gradio['name2']], 
        outputs=None
    ).then(
            fn=utils.gather_interface_values, 
        inputs=utils.gradget(shared.input_elements), 
        outputs=utils.gradget('interface_state')
    ).then(
            fn=chat.generate_chat_reply_wrapper, 
        inputs=utils.gradget(chat_inputs), 
        outputs=utils.gradget('display', 'history'), 
        show_progress=False
    ).then(
        lambda: None, None, picture_select, show_progress=False
    )
