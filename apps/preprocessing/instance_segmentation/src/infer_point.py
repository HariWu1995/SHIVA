import numpy as np

from .utils import load_model, SamPredictor


def inference_prompt(
        image, 
        model, 
        points_coord,
        points_label,
    ):
    segmentor = load_model(model)
    predictor = SamPredictor(segmentor)
    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=np.array(points_coord),
        point_labels=np.array(points_label),
        multimask_output=True,
    )
    return masks


def detection_prompt():
    pass

