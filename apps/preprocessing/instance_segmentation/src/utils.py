import os
from unittest.mock import patch

from pathlib import Path
from typing import Union, Any, Tuple, Dict

import torch
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM as AutoDetector
from transformers import OwlViTProcessor, OwlViTForObjectDetection as OwlViTDetector
from transformers.dynamic_module_utils import get_imports

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator as SamMaskGenerator


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_ROOT = os.environ.get('SHIVA_CKPT_ROOT', None)
if CHECKPOINT_ROOT is not None:
    SAM_DIR = Path(CHECKPOINT_ROOT) / 'sam'
    OWL_DIR = Path(CHECKPOINT_ROOT) / 'owlvit-base-patch32'
    FLR_DIR = Path(CHECKPOINT_ROOT) / 'Florence-2-base'
    YLW_DIR = Path(CHECKPOINT_ROOT) / 'yolo'
else:
    SAM_DIR = Path(__file__).parents[4] / 'checkpoints/sam'
    OWL_DIR = Path(__file__).parents[4] / 'checkpoints/owlvit-base-patch32'
    FLR_DIR = Path(__file__).parents[4] / 'checkpoints/Florence-2-base'
    YLW_DIR = Path(__file__).parents[4] / 'checkpoints/yolo'

OWL_DIR = str(OWL_DIR)
if os.path.isdir(OWL_DIR) is False:
    OWL_DIR = "google/owlvit-base-patch32"

FLR_DIR = str(FLR_DIR)
if os.path.isdir(FLR_DIR) is False:
    FLR_DIR = "microsoft/Florence-2-base"

sam_models = {
    'vit_b': str(SAM_DIR / 'sam_vit_b_01ec64.pth'),
    'vit_l': str(SAM_DIR / 'sam_vit_l_0b3195.pth'),
    'vit_h': str(SAM_DIR / 'sam_vit_h_4b8939.pth'),
}

det_models = {
     'owl-vit': OWL_DIR,
    'florence': FLR_DIR,
}

yolow_path = YLW_DIR / "yolov8l-world.pt"
if os.path.isfile(yolow_path):
    det_models['yolo-world'] = yolow_path


def load_model(model_type, device=DEVICE):
    segmentor = sam_model_registry[model_type](checkpoint=sam_models[model_type]).to(device)
    return segmentor


def load_detector(model_type, device=DEVICE):

    if model_type == "owl-vit":
        processor = OwlViTProcessor.from_pretrained(OWL_DIR)
        detector = OwlViTDetector.from_pretrained(OWL_DIR).to(device).eval()
        return processor, detector

    elif model_type == "florence":
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            processor = AutoProcessor.from_pretrained(FLR_DIR, trust_remote_code=True)
            detector = AutoDetector.from_pretrained(FLR_DIR, trust_remote_code=True).to(device).eval()
        return processor, detector

    elif model_type == "yolo-world":
        detector = YOLO(yolow_path)
        return None, detector


def fixed_get_imports(filename: Union[str, os.PathLike]) -> list[str]:
    """
    Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72
    """
    if str(filename).endswith("modeling_florence2.py") is False:
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

