"""
Reference:
    https://huggingface.co/microsoft/Florence-2-base
    https://huggingface.co/spaces/SkalskiP/florence-sam/tree/main
"""
from PIL import Image
from PIL.Image import Image as ImageClass
from typing import Union, Any, Tuple, Dict

import supervision as sv
import torch

from .utils import DEVICE


def detection(
        detector: Any,
        processor: Any,
        image: ImageClass,
        text: str = "",
        thresh: float = 0.1,
        device: torch.device = DEVICE,
    ) -> Tuple[str, Dict]:

    if not isinstance(image, ImageClass):
        image = Image.fromarray(image)
    
    detector.set_classes([text])
    result = detector.predict(image)[0]

    # Format
    detections = sv.Detections.from_ultralytics(result)
    boxes = detections.xyxy

    return boxes


