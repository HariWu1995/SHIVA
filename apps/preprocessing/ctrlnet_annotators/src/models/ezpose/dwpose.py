import os
from typing import Callable, Dict, Optional, Union

import cv2
import PIL
import PIL.Image

import numpy as np
import torch

from .body_estimation import Wholebody, resize_image
from .format import format_openpose
from .draw import draw_openpose


class DWposeDetector:

    def __init__(
            self, 
            pose_model_path: str,
            det_model_path: str | None = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ):
        self.use_det = det_model_path is not None
        self.pipeline = Wholebody(
            model_det = det_model_path, 
            model_pose = pose_model_path, 
                device = device,
        )

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray],
        resolution: int = 512,
        output_type: str = "pil",
        bboxes = None,
        # return_mmpose: bool = False,
        draw_pose: bool = True,
        **drawkwargs,
    ) -> Union[PIL.Image.Image, np.ndarray, Dict]:
        if type(image) != np.ndarray:
            image = np.array(image.convert("RGB"))

        image = image.copy()
        original_height, original_width, _ = image.shape

        if resolution > 0:
            image = resize_image(image, target_resolution=resolution)
            height, width, _ = image.shape
        else:
            height = original_height
            width = original_width

        candidates, scores = self.pipeline(image, bboxes, return_mmpose)

       # TODO: format & draw MMPose
        # if return_mmpose:
        #     return candidates, scores

        pose = format_openpose(candidates, scores, width, height)

        if not draw_pose:
            return pose

        pose_image = draw_openpose(pose, height=height, width=width, **drawkwargs)
        pose_image = cv2.resize(pose_image, (original_width, original_height), cv2.INTER_LANCZOS4)

        if output_type == "pil":
            pose_image = PIL.Image.fromarray(pose_image)
        elif output_type == "np":
            pass
        else:
            raise ValueError("output_type should be 'pil' or 'np'")

        return pose_image
