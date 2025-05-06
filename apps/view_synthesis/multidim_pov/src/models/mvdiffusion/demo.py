import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from pathlib import Path
current_workdir = Path(__file__).resolve().parent

import yaml
import argparse
from datetime import datetime

import cv2
from PIL import Image

import numpy as np
import torch
torch.manual_seed(0)

from .src.lightning_pano_gen      import PanoGenerator
from .src.lightning_pano_outpaint import PanoOutpaintGenerator as PanoOutpaintor
from .tools.pano_video_generation import generate_video

from .utils import preprocess_image, multiview_Rs_Ks
from ..path import MVDIFF_LOCAL_MODELS
from ... import shared


def load_config(config_path):
    config = yaml.load(open(config_path, 'rb'), Loader=yaml.SafeLoader)
    return config


def load_model(config, ckpt_path, outpaint: bool = False):
    model_class = PanoOutpaintor if outpaint else PanoGenerator
    model = model_class(config)
    model.load_state_dict(
        torch.load(ckpt_path, map_location='cpu')['state_dict'], strict=True)
    model.eval()
    model.to(device=shared.device)
    return model


if __name__ == "__main__":

    prompt = 'a photorealistic rendering of a stage with a three-tiered circular platform, '\
             'the platform is light purple and white, '\
             'a backdrop of deep purple velvet curtains elegantly draped, '\
             'three hanging spotlights illuminating the stage, '\
             'a large circular background element that is a peachy-orange gradient, '\
             'pink background, clean and modern aesthetic'

    # T2pano
    config_path = str(current_workdir / 'configs/pano_generation.yaml')
    ckpt_path = MVDIFF_LOCAL_MODELS["sd2_original/mvdiffusion"]
    outpaint = False
    image = None

    # T+I2pano
    # outpaint = True
    # config_path = str(current_workdir / 'configs/pano_outpainting.yaml')
    # ckpt_path = MVDIFF_LOCAL_MODELS["sd2_outpaint/mvdiffusion"]
    # image_path = "C:/Users/Mr. RIAH/Pictures/_stage/stage-06.png"
    # image = cv2.imread(image_path)
    # image = preprocess_image(image, config['dataset']['resolution'])
    # image = torch.tensor(image).to(device=shared.device)

    # Load model
    config = load_config(config_path)
    model = load_model(config, ckpt_path, outpaint)

    # Pipeline
    resolution = config['dataset']['resolution']
    num_views = 8

    images = torch.zeros((1, num_views, resolution, resolution, 3)).to(device=shared.device)
    if image is not None:
        images[0, 0] = image

    prompt = [prompt] * num_views

    Rs, Ks = multiview_Rs_Ks(resolution, num_views)
    Ks = torch.tensor(Ks).to(device=shared.device)[None]
    Rs = torch.tensor(Rs).to(device=shared.device)[None]

    batch = {
        'images': images,
        'prompt': prompt,
             'R': Rs,
             'K': Ks,
    }

    generated = model.inference(batch)

    # Save output
    prefix = 'mvdiff_' + ('outpaint' if outpaint else 'generate')
    image_paths = []
    for i in range(num_views):
        image_path = f"./temp/{prefix}_view{i+1:02d}.png"
        image_paths.append(image_path)
        im = Image.fromarray(generated[0, i])
        im.save(image_path)

    generate_video(image_paths, './temp', prefix)
