import numpy as np
import torch

from .utils import load_model, SamPredictor, load_detector, DEVICE
from .detect_owlvit import detection as detect_owlvit
from .detect_florence import detection as detect_florence
from .detect_yoloworld import detection as detect_yoloworld


def inference_prompt(
        image, 
        prompt,
        sam_model,
        det_model, 
        det_thresh: float = 0.1,
    ):
    # Load SAM
    segmentor = load_model(sam_model)
    predictor = SamPredictor(segmentor)
    predictor.set_image(image)

    # Load open-vocab Detector
    processor, detector = load_detector(det_model)
    if det_model == 'florence':
        detection_fn = detect_florence 
    elif det_model == 'owl-vit':
        detection_fn = detect_owlvit
    elif det_model == "yolo-world":
        detection_fn = detect_yoloworld

    anns = []
    texts = [t.strip() for t in prompt.split(",")]
    for text in texts:

        # Detection -> bbox: xyxy
        bboxes = detection_fn(detector, processor, image, text, det_thresh)
        if len(bboxes) == 0:
            print(f'[WARNING] Cannot detect {text}')
            continue
        bbox = bboxes[0]

        # Segmentation with box prompt
        # bbox = torch.Tensor(bbox).to(DEVICE)
        # bbox = predictor.transform.apply_boxes_torch(bbox, image.shape[:2])
        mask, *_ = predictor.predict(box=bbox, multimask_output=False)
        anns.append([mask.astype(float), text])

    return anns


