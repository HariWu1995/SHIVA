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


FLORENCE_TASK = {
                'OBJECT_DETECTION': '<OD>',
            'OPEN_VOCAB_DETECTION': '<OPEN_VOCABULARY_DETECTION>',
                         'CAPTION': '<MORE_DETAILED_CAPTION>',
            'DENSE_REGION_CAPTION': '<DENSE_REGION_CAPTION>',
     'CAPTION_TO_PHRASE_GROUNDING': '<CAPTION_TO_PHRASE_GROUNDING>',
}


def detection(
        detector: Any,
        processor: Any,
        image: ImageClass,
        text: str = "",
        thresh: float = 0.1,
        device: torch.device = DEVICE,
        task: str = FLORENCE_TASK['OPEN_VOCAB_DETECTION'],
    ) -> Tuple[str, Dict]:

    if not isinstance(image, ImageClass):
        image = Image.fromarray(image)
    
    prompt = task + text
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = detector.generate(input_ids=inputs["input_ids"],
                                    pixel_values=inputs["pixel_values"],
                                   max_new_tokens=1024,
                                        num_beams=3)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(generated_text, task=task, image_size=image.size)

    # Format
    detections = sv.Detections.from_lmm(lmm=sv.LMM.FLORENCE_2, result=result, resolution_wh=image.size)
    boxes = detections.xyxy

    return boxes


