from copy import deepcopy
from pathlib import Path
from datetime import datetime

import torch
import gradio as gr

from .. import shared as ui


symbols = dict(
    trigger = '🚀',
    refresh = '🔄',
    release = '🗑️',
    delete = '🗑️',
    save = '💾',
)


def current_time():
    return f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"


# Helper function to get multiple values from ui.gradio
def gradget(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]
    return [ui.gradio[k] for k in keys if k in ui.gradio.keys()]


#################################################
#               Wrapper Chatbox                 #
#################################################

CHAT_COLOR = dict(
    narrator = 'powderblue',
    player_1 = 'pink',
    player_2 = 'tomato',
     unknown = 'gray',
)

BUBBLE_CHAT_TEMPLATE = """
<div style="display: flex; gap: 5px;">
  <div style="background-color: {color}; padding: 10px; border-radius: 10px;">
    <div> [{role}] </div> 
    <div>  {text}  </div> 
  </div>
</div>
"""

def colorize_bubble_chat(content: str, role: str, role_class: str):
    if role_class not in CHAT_COLOR.keys():
        role_class = 'unknown'
    color = CHAT_COLOR.get(role_class, 'gray')
    chat = deepcopy(BUBBLE_CHAT_TEMPLATE)
    return chat.format(color=color, role=role, text=content)

