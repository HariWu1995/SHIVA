import os, sys
import argparse
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image

import numpy as np
import torch


#################################
#           Pipeline            #
#################################

from .. import shared
from ..utils import set_seed, clear_torch_cache

# Link to model
current_workdir = Path(__file__).resolve().parent
sys.path.append(str(current_workdir))

from mvdream.model_zoo import build_model
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models import DDIMSampler
from mvdream.camera_utils import get_camera


def load_model(model_name: str):
    model_ver = model_name.split('/')[0]

    model_path = shared.MVDIFF_LOCAL_MODELS[model_name]
    config_path = current_workdir / f"mvdream/configs/{model_ver}.yaml"

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(
        torch.load(model_path, map_location='cpu'), strict=False)

    model.device = shared.device
    model.to(device=shared.device)
    model.eval()
    return model


def inference(
    model, 
    prompt: str | list, 
    image_size: int = 256, 
    diffusion_steps: int = 20, 
    guidance_scale: float = 7.5, 
    batch_size: int = 4, 
    num_frames: int = 1,
    ddim_eta: float = 0., 
    **camera_kwargs
):
    if isinstance(prompt, (list, tuple)) is False:
        prompt = [prompt]

    B = batch_size
    shape = [4, image_size // 8, image_size // 8]

    sampler = DDIMSampler(model)

    # pre-compute camera matrices
    if  camera_kwargs.get('camera_elev', None) is not None \
    and camera_kwargs.get('camera_azim', None) is not None \
    and camera_kwargs.get('camera_azim_span', None) is not None:
        camera = get_camera(num_frames, elevation=camera_kwargs['camera_elev'], 
                                    azimuth_start=camera_kwargs['camera_azim'], 
                                     azimuth_span=camera_kwargs['camera_azim_span'])
        camera = camera.repeat(B // num_frames, 1).to(shared.device)
    else:
        camera = None

    with torch.no_grad(), \
        torch.autocast(device_type=str(shared.device), dtype=shared.dtype):

        c = model.get_learned_conditioning(prompt).to(shared.device)
        uc = model.get_learned_conditioning([""]).to(shared.device)

        c_ = {"context": c.repeat(B, 1, 1)}
        uc_ = {"context": uc.repeat(B, 1, 1)}

        if camera is not None:
            c_["camera"] = camera
            uc_["camera"] = camera
            c_["num_frames"] = num_frames
            uc_["num_frames"] = num_frames

        samples_ddim, _ = sampler.sample(
                            S=diffusion_steps, 
                            conditioning=c_,
                            batch_size=B, 
                            shape=shape,
                            verbose=False, 
                            unconditional_guidance_scale=guidance_scale,
                            unconditional_conditioning=uc_,
                            eta=ddim_eta, 
                            x_T=None
                        )

        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0, 2, 3, 1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":

    model_name = "sd15/mvdream"
    # model_name = "sd21/mvdream"

    model = load_model(model_name)
    # print(model)

    set_seed(19_05_1995)

    prompt = "3d asset, an astronaut riding a horse"
    gen_params = dict(
            image_size = 256, 
        diffusion_steps = 25, 
        guidance_scale = 7.5,
            batch_size = 4, 
            num_frames = 4,
    )

    cam_params = dict(
        camera_elev = 15,
        camera_azim = 90,
        camera_azim_span = 360,
    )

    num_tests = 3
    for i in range(num_tests):
        imv = inference(model, prompt, **gen_params, **cam_params)
        img = np.concatenate(imv, axis=1)
        img = Image.fromarray(img)
        img.save(f"./temp/{model_name.replace('/','_')}_{i:02d}.png")

