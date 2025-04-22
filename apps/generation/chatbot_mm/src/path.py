import os
from pathlib import Path

import torch
import numpy as np
import cv2

from .utils import get_device_memory
from .default import INSTRUCTION_DIR, CHARACTER_DIR, GRAMMAR_DIR, PROMPT_DIR, PRESET_DIR


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_ROOT = os.environ.get('SHIVA_CKPT_ROOT', None)
if CHECKPOINT_ROOT is not None:
    MULTIMODAL_LM_DIR = Path(CHECKPOINT_ROOT) / 'chatbot'
else:
    MULTIMODAL_LM_DIR = Path(__file__).parents[4] / 'checkpoints/chatbot'

if os.path.isdir(MULTIMODAL_LM_DIR) is False:
    os.makedirs(MULTIMODAL_LM_DIR)


MLM_REMOTE_MODELS = {
      "qwen-2.5-7b-instruct": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5VL-7b-instruct": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
     "llama-3.2-1b-instruct": "https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/resolve/main/llama-3.2-1b-instruct-q8_0.gguf",
      "deepseek-7b-chat"    : "https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat",
      "deepseek-r1-1b5-qwen": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "moondream-v2"      : "https://huggingface.co/vikhyatk/moondream2",
}


MLM_LOCAL_MODELS = {
      "deepseek-r1-1b5-qwen": str(MULTIMODAL_LM_DIR / "deepseek-r1-1b5-qwen"),      # 3.6Gb
      "deepseek-7b-chat"    : str(MULTIMODAL_LM_DIR / "deepseek-7b-chat"),          # 14Gb
      "qwen-2.5-7b-instruct": str(MULTIMODAL_LM_DIR / "qwen-2.5-7b-instruct"),      # 16Gb
    "qwen-2.5VL-7b-instruct": str(MULTIMODAL_LM_DIR / "qwen-2.5vl-7b-instruct"),    # 17Gb
     "llama-3.2-1b-instruct": str(MULTIMODAL_LM_DIR / "llama-3.2-1b-q8-instruct"),
        "moondream-v2"      : str(MULTIMODAL_LM_DIR / "moondream_v2"),
}


MULTIMODAL_ALLOWED = {
      "deepseek-r1-1b5-qwen": ['text'],
      "deepseek-7b-chat"    : ['text'],
     "llama-3.2-1b-instruct": ['text'],
      "qwen-2.5-7b-instruct": ['text'],
    "qwen-2.5VL-7b-instruct": ['text', 'image', 'video'],
        "moondream-v2"      : ['text', 'image'],
}


device_vram_Gb = get_device_memory(device_id=0, device_type='cuda') / 1024

if device_vram_Gb < 24:
    del MLM_LOCAL_MODELS["qwen-2.5-7b-instruct"]
    del MLM_LOCAL_MODELS["qwen-2.5VL-7b-instruct"]

if device_vram_Gb < 16:
    del MLM_LOCAL_MODELS["deepseek-7b-chat"]

if device_vram_Gb < 4:
    del MLM_LOCAL_MODELS["deepseek-r1-1b5-qwen"]

if device_vram_Gb < 3:
    del MLM_LOCAL_MODELS["llama-3.2-1b-instruct"]


LORA_LOCAL_MODELS = {}
LORA_REMOTE_MODELS = {}


def load_file_from_url(
    remote_url: str,
    model_dir: str | None = None,
    local_path: str | None = None,
    hash_prefix: str | None = None,
    progress: bool = True,
) -> str:
    raise NotImplementedError()


if os.environ.get('SHIVA_CKPT_PRELOAD', False):
    for model_name, model_path in MLM_LOCAL_MODELS.items():
        if os.path.isfile(model_path):
            continue
        load_file_from_url(remote_url=MLM_REMOTE_MODELS[model_name], 
                           local_path=MLM_LOCAL_MODELS[model_name])

