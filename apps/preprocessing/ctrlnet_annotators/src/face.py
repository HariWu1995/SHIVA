import os
import cv2
import numpy as np

import torch
import torchvision as tv
from einops import rearrange

from .utils import DEVICE, CTRL_ANNOT_LOCAL_MODELS, CTRL_ANNOT_REMOTE_MODELS, load_file_from_url


#####################################
#           MediaPipe-Face          #
#####################################
from .models.mediapipe_face import (
    IS_MEDIAPIPE_INSTALLED, 
    generate_annotation as mediapipe_annotation,
)


#############################
#       Universal           #
#############################
all_options_face = []
if IS_MEDIAPIPE_INSTALLED:
    all_options_face.append("mediapipe")

def apply_face(input_image, model: str = "mediapipe", **kwargs):

    if model == "mediapipe": 
        if not IS_MEDIAPIPE_INSTALLED:
            print("MediaPipe is not install")
            return input_image
        return mediapipe_annotation(input_image, **kwargs)

    else:
        raise ValueError(f"model = {model} is not supported!")
    

