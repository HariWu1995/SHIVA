import os

import torch
from torchvision import transforms as vTransforms
from einops import rearrange

from PIL import Image
import cv2
import numpy as np

from .utils import DEVICE, CTRL_ANNOT_LOCAL_MODELS, CTRL_ANNOT_REMOTE_MODELS, load_file_from_url


#########################################
#           Animal-Face                 #
#########################################
from .models.animeface import UNet as AnimeFaceUNet

class AnimeFaceSegment:

    model_name = "anifaceseg"

    COLOR_BACKGROUND = (255, 255, 0)
    COLOR_HAIR = (0, 0, 255)
    COLOR_EYE = (255, 0, 0)
    COLOR_MOUTH = (255, 255, 255)
    COLOR_FACE = (0, 255, 0)
    COLOR_SKIN = (0, 255, 255)
    COLOR_CLOTHES = (255, 0, 255)
    COLOR_PALETTE = [
        COLOR_BACKGROUND, 
        COLOR_HAIR, COLOR_EYE, COLOR_MOUTH, 
        COLOR_FACE, COLOR_SKIN, COLOR_CLOTHES
    ]

    def __init__(self, device = DEVICE, preload: bool = False):
        self.device = device
        self.model = None
        self.prep = vTransforms.Compose([  
                    vTransforms.Resize(512, interpolation=vTransforms.InterpolationMode.BICUBIC),  
                    vTransforms.ToTensor(),
                ])
        if preload:
            self.load_model()

    def load_model(self):
        local_model_path = CTRL_ANNOT_LOCAL_MODELS[self.model_name]
        if not os.path.exists(local_model_path):
            remote_model_path = CTRL_ANNOT_REMOTE_MODELS[self.model_name]
            load_file_from_url(remote_url=remote_model_path, 
                                local_file=local_model_path)
        net = AnimeFaceUNet()
        ckpt = torch.load(local_model_path, map_location=self.device)
        for key in list(ckpt.keys()):
            if 'module.' in key:
                ckpt[key.replace('module.', '')] = ckpt[key]
                del ckpt[key]
        net.load_state_dict(ckpt)
        net.eval()
        self.model = net.to(self.device)

    def offload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, img: np.ndarray | Image.Image):

        if self.model is None:
            self.load_model()

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        with torch.no_grad():
            img = self.prep(img).unsqueeze(dim=0).to(self.device)
            seg = self.model(img).squeeze(dim=0)
            seg = seg.cpu().detach().numpy()

        out = rearrange(seg, 'h w c -> w c h')
        out = [[self.COLOR_PALETTE[np.argmax(val)] for val in buf] for buf in out]
        out = np.array(out).astype(np.uint8)
        return out


#############################
#       Universal           #
#############################
all_options_edge = ["aniface"]

def apply_segment(input_image, model: str = "aniface", *, **kwargs):

    if model == "aniface":
        segmentor = AnimeFaceSegment()
    return segmentor(input_image)

    else:
        raise ValueError(f"model = {model} is not supported!")

