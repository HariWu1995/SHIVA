import gradio as gr
import pandas as pd

from . import shared as ui


# Helper function to get multiple values from ui.gradio
def gradget(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]
    return [ui.gradio[k] for k in keys if k in ui.gradio.keys()]


def fill_prompts_mv(prompts):
    prompts = pd.Series(prompts)
    prompts = prompts.ffill().bfill()
    return prompts.tolist()

