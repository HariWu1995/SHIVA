import html
import random
import torch
from transformers import is_torch_npu_available, is_torch_xpu_available


def set_manual_seed(seed: str | int = -1):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 2**31)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif is_torch_xpu_available():
        torch.xpu.manual_seed_all(seed)
    elif is_torch_npu_available():
        torch.npu.manual_seed_all(seed)

    return seed


DARK_YELLOW = "\033[38;5;3m"
RESET = "\033[0m"

def print_prompt(prompt, max_chars=2_000):

    if len(prompt) > max_chars:
        half_chars = max_chars // 2
        hidden_len = len(prompt[half_chars:-half_chars])
        hidden_msg = f"{DARK_YELLOW}[...{hidden_len} characters hidden...]{RESET}"
        print(prompt[:half_chars] + hidden_msg + prompt[-half_chars:])
    else:
        print(prompt)


from ...grui.html import generate_basic_html

def formatted_outputs(reply, model_name):
    return html.unescape(reply), generate_basic_html(reply)

