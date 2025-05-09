"""
Reference:
    https://huggingface.co/google/owlvit-base-patch32
"""
from PIL import Image
from typing import Union, Any, Tuple, Dict
import torch

from .utils import DEVICE


def detection(
        detector: Any,
        processor: Any,
        image: Image.Image,
        text: str = "",
        thresh: float = 0.01, 
        device: torch.device = DEVICE,
    ):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    inputs = processor(text=[[text]], images=image, return_tensors="pt").to(device)
    outputs = detector(**inputs)

    # Target size (height, width) to rescale box predictions [batch_size, 2]
    # target_size = torch.Tensor([image.shape[:2]]).to(device)
    target_size = torch.Tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs=outputs, 
                                                 target_sizes=target_size,
                                                    threshold=thresh)

    # get the box with best score
    # scores = torch.sigmoid(outputs.logits)
    # best_scores, best_idxs = torch.topk(scores, k=1, dim=1)
    # best_idxs = best_idxs.squeeze(1).tolist()

    # Retrieve predictions for the 1st image corresponding to text queries
    i = 0
    # scores = results[i]["scores"]
    # labels = results[i]["labels"]
    boxes = results[i]["boxes"]
    boxes = boxes.cpu().detach().numpy()
    # boxes = boxes[np.newaxis, :, :]
    return boxes

