import os
from pathlib import Path

import torch
import numpy as np
import cv2


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_ROOT = os.environ.get('SHIVA_CKPT_ROOT', None)
if CHECKPOINT_ROOT is not None:
    MULTIMODAL_LM_DIR = Path(CHECKPOINT_ROOT) / 'chatbot'
else:
    MULTIMODAL_LM_DIR = Path(__file__).parents[4] / 'checkpoints/chatbot'

if os.path.isdir(MULTIMODAL_LM_DIR) is False:
    os.makedirs(MULTIMODAL_LM_DIR)


MLM_LOCAL_MODELS = {
    "llava-7b-4bit": str(MULTIMODAL_LM_DIR / "llava-7b-4bit"),
    "llama-7b-4bit": str(MULTIMODAL_LM_DIR / "llama-7b-4bit"),
      "deepseek-7b": str(MULTIMODAL_LM_DIR / "deepseek-7b"),
      "qwen-2.5-7b": str(MULTIMODAL_LM_DIR / "qwen-2.5-7b"),
    "qwen-2.5VL-7b": str(MULTIMODAL_LM_DIR / "qwen-2.5vl-7b"),
}

MLM_REMOTE_MODELS = {
    "llava-7b-4bit": "https://huggingface.co/wojtab/llava-7b-v0-4bit-128g",
    "llama-7b-4bit": "https://huggingface.co/kuleshov/llama-7b-4bit",
      "deepseek-7b": "https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat",
      "qwen-2.5-7b": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5VL-7b": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
}


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