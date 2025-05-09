import numpy as np

from .utils import load_model, SamMaskGenerator


def inference_auto(
        image, 
        model, 
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        box_nms_thresh: float = 0.7,
        crop_nms_thresh: float = 0.7,
    ):
    segmentor = load_model(model)
    segment_kwargs = dict(
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        box_nms_thresh=box_nms_thresh,
        crop_nms_thresh=crop_nms_thresh,
            output_mode="binary_mask",
    )
    mask_generator = SamMaskGenerator(segmentor, **segment_kwargs)

    assert isinstance(image, np.ndarray)
    masks = mask_generator.generate(image)
    return masks

