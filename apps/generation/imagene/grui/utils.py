from copy import deepcopy
from pathlib import Path
from datetime import datetime

import gradio as gr

from . import shared as ui


symbols = dict(
    trigger = '🚀 Trigger',
    refresh = '🔄 Refresh',
    reset = '🔄 Reset',
    release = '🗑️ Release',
    delete = '🗑️ Delete',
    offload = '⌛ Offload',
    save = '💾 Save',
    undo = '🔙 Undo',
)


def current_time():
    return f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"


# Helper function to get multiple values from ui.gradio
def gradget(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]
    return [ui.gradio[k] for k in keys if k in ui.gradio.keys()]


# Helper functions to manipulate prompts
from ..src.default import POSITIVE_PROMPT, NEGATIVE_PROMPT, STYLES_PROMPT

def extend_prompt(prompt, xrompt):
    if xrompt in prompt:
        return prompt
    if len(prompt) <= 0:
        return xrompt
    if prompt[-1].isalpha():
        prompt = prompt + ', ' + xrompt
    else:
        prompt = prompt + ' ' + xrompt
    return prompt


def update_prompt(prompt):
    return extend_prompt(prompt, POSITIVE_PROMPT)


def update_nrompt(prompt):
    return extend_prompt(prompt, NEGATIVE_PROMPT)


def apply_styles(prompt, styles):
    if not styles:
        return prompt
    for style in styles:
        srompt = STYLES_PROMPT.get(style, "").lower()
        print(style)
        print(srompt)
        if srompt == "":
            continue
        if srompt in prompt:
            continue
        if len(prompt) <= 0:
            prompt = srompt
        elif prompt[-1].isalpha():
            prompt = prompt + '. ' + srompt
        else:
            prompt = prompt + ' ' + srompt
    return prompt

