import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torchvision import transforms
from einops import rearrange

from .utils import DEVICE, CTRL_ANNOT_LOCAL_MODELS, CTRL_ANNOT_REMOTE_MODELS, load_file_from_url


#########################################################################
#                   LeRes / LeRes++                                     #
#   https://github.com/Mikubill/sd-webui-controlnet/discussions/1141    #
#   https://github.com/compphoto/BoostingMonocularDepth                 #
#########################################################################
from .models.leres import estimate_leres, estimate_boost, RelDepthModel, normalize_moduledict
from .models.pix2pix import Pix2PixOptions, Pix2Pix4DepthModel


class LeResEstimate:

    model_type = 0
    model_name = "leres"
    p2p_name = "pix2pix"

    def __init__(self, device = DEVICE, boost: bool = False, preload: bool = False):
        self.device = device
        self.boost = boost
        self.model = None
        self.p2p_model = None
        if preload:
            self.load_model()

    def load_model(self):
        # Load LeRes
        local_model_path = CTRL_ANNOT_LOCAL_MODELS[self.model_name]
        if not os.path.exists(local_model_path):
            remote_model_path = CTRL_ANNOT_REMOTE_MODELS[self.model_name]
            load_file_from_url(remote_url=remote_model_path, 
                                local_file=local_model_path)
        checkpoint = torch.load(local_model_path, map_location='cpu')
        checkpoint = normalize_moduledict(checkpoint['depth_model'], prefix="module.")  # remove multi-GPU prefix
        model = RelDepthModel(backbone='resnext101')
        model.load_state_dict(checkpoint, strict=True)
        self.model = model.to(device=device).eval()

        if not self.boost:
            return

        # Load Pix2Pix
        local_model_path = CTRL_ANNOT_LOCAL_MODELS[self.p2p_name]
        if not os.path.exists(local_model_path):
            remote_model_path = CTRL_ANNOT_REMOTE_MODELS[self.p2p_name]
            load_file_from_url(remote_url=remote_model_path, 
                                local_file=local_model_path)
        p2p_opts = Pix2PixOptions().parse()
        if not torch.cuda.is_available():
            p2p_opts.gpu_ids = []
        p2p_model = Pix2Pix4DepthModel(p2p_opts, model_path=local_model_path, device=device)
        p2p_model.load_networks('pix2pix')
        p2p_model.eval()
        self.p2p_model = p2p_model

    def offload_model(self):
        if self.model is not None:
            self.model.cpu()
        if self.p2p_model is not None:
            self.p2p_model = self.p2p_model.unload_network('G')

    def __call__(self, input_image: np.ndarray | Image.Image, thresh_a=0, thresh_b=0):

        if self.model is None or \
        self.p2p_model is None:
            self.load_model()

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        assert input_image.ndim == 3

        height, width, dim = input_image.shape

        with torch.no_grad():
            if boost:
                depth = estimate_boost(input_image, self.model, self.model_type, self.p2p_model, max(width, height))
            else:
                depth = estimate_leres(input_image, self.model, width, height)

        num_bytes = 2
        depth_min = depth.min()
        depth_max = depth.max()
        max_val = (2 ** (8*num_bytes)) - 1

        # check output before normalizing and mapping to 16 bit
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape)
        
        # single channel, 16 bit image
        depth_image = out.astype("uint16")

        # convert to uint8
        alpha = 255.0 / 65535.0
        depth_image = cv2.convertScaleAbs(depth_image, alpha=alpha)

        # remove near
        if thresh_a != 0:
            thresh_a = (thresh_a / 100) * 255
            depth_image = cv2.threshold(depth_image, thresh_a, 255, cv2.THRESH_TOZERO)[1]

        # invert image
        depth_image = cv2.bitwise_not(depth_image)

        # remove bg
        if thresh_b != 0:
            thresh_b = (thresh_b / 100) * 255
            depth_image = cv2.threshold(depth_image, thresh_b, 255, cv2.THRESH_TOZERO)[1]

        return depth_image


#########################################
#               MiDaS                   #
#   https://github.com/isl-org/MiDaS    #
#########################################
from .models.midas import DPTDepthModel, MidasNet, MidasNet_small, midas_transform, midas_backbones


class MiDaSEstimate:
    
    def __init__(self, model_name, device = DEVICE, preload: bool = False):
        assert model_name in ["dpt_large","dpt_hybrid","midas_v21","midas_v21s"], \
            f"{model_name} is not supported yet!"
        self.model_name = model_name
        self.device = device
        self.model = None
        if preload:
            self.load_model()

    def load_model(self):
        local_model_path = CTRL_ANNOT_LOCAL_MODELS[self.model_name]
        if not os.path.exists(local_model_path):
            remote_model_path = CTRL_ANNOT_REMOTE_MODELS[self.model_name]
            load_file_from_url(remote_url=remote_model_path, 
                                local_file=local_model_path)

        if self.model_name in ["dpt_large", "dpt_hybrid"]:
            model = DPTDepthModel(
                        path = local_model_path,
                    backbone = midas_backbones[self.model_name],
                non_negative = True,
            )
        
        elif self.model_name == "midas_v21":
            model = MidasNet(path=local_model_path, non_negative=True)
        
        elif self.model_name == "midas_v21s":
            model = MidasNet_small(
                    path = local_model_path, 
                backbone = midas_backbones[self.model_name],
                features = 64, 
              exportable = True,
            non_negative = True, 
                  blocks = {'expand': True},
            )
        
        model.eval()
        self.model = model.to(self.device)
        self.transform = midas_transform(self.model_name)

    def offload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, 
        input_image, 
        z_value: float = np.pi * 2.0, 
        bg_thresh: float = 0.1, 
        use_normmap: bool = True,
    ):
        if self.model is None:
            self.load_model()

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        assert input_image.ndim == 3

        with torch.no_grad():
            image = torch.from_numpy(input_image).float()
            image = image.to(self.device)
            image = image / 127.5 - 1.0
            image = rearrange(image, 'h w c -> 1 c h w')
            depth = model(image)[0]

            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
        
        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)
        if not use_normmap:
            return depth_image
        
        depth_np = depth.cpu().numpy()
        
        # Normal Map Calculation
        x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
        y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
        z = np.ones_like(x) * z_value
        
        x[depth_pt < bg_thresh] = 0
        y[depth_pt < bg_thresh] = 0

        normal = np.stack([x, y, z], axis=2)
        normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
        normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)[:, :, ::-1]

        return normal_image


#############################################
#               NormalBae                   #
#############################################
from .models.normalbae import NormalNet, NormalConfig


class NormalBaeEstimate:

    model_name = 'normalbae'

    def __init__(self, device = DEVICE, preload: bool = False):
        self.device = device
        self.model = None
        if preload:
            self.load_model()

    def load_model(self):
        local_model_path = CTRL_ANNOT_LOCAL_MODELS[self.model_name]
        if not os.path.exists(local_model_path):
            remote_model_path = CTRL_ANNOT_REMOTE_MODELS[self.model_name]
            load_file_from_url(remote_url=remote_model_path, 
                                local_file=local_model_path)

        model = NormalNet(NormalConfig)
        ckpt = torch.load(local_model_path, map_location='cpu')['model']
        state_dict = {}
        for k, v in ckpt.items():
            if k.startswith('module.'):
                k_ = k.replace('module.', '')
                state_dict[k_] = v
            else:
                state_dict[k] = v
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model.to(self.device)
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def offload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image):
        if self.model is None:
            self.load_model()

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        assert input_image.ndim == 3

        with torch.no_grad():
            image = torch.from_numpy(input_image).to(self.device)
            image = image.float() / 255.0
            image = rearrange(image, 'h w c -> 1 c h w')
            image = self.transform(image)

            normal = self.model(image)
            normal = normal[0][-1][:, :3]
            normal = ((normal + 1) * 0.5).clip(0, 1)
            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()

        normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)
        return normal_image
        return normal_image


#############################################
#                   ZoeDepth                #
#   https://github.com/isl-org/ZoeDepth     #
#############################################
from .models.zoe import ZoeDepth, get_config as get_zoe_config

class ZoeDepthEstimate:

    model_name = "zoe"

    def __init__(self, device = DEVICE, preload: bool = False):
        self.device = device
        self.model = None
        if preload:
            self.load_model()

    def load_model(self):
        local_model_path = CTRL_ANNOT_LOCAL_MODELS[self.model_name]
        if not os.path.exists(local_model_path):
            remote_model_path = CTRL_ANNOT_REMOTE_MODELS[self.model_name]
            load_file_from_url(remote_url=remote_model_path, 
                                local_file=local_model_path)

        conf = get_zoe_config("zoedepth", "infer")
        model = ZoeDepth.build_from_config(conf)
        model.load_state_dict(torch.load(local_model_path, map_location=self.device)['model'])
        model.eval()
        self.model = model.to(self.device)

    def offload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image):
        if self.model is None:
            self.load_model()

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        assert input_image.ndim == 3

        with torch.no_grad():
            image = torch.from_numpy(input_image).float().to(self.device)
            image = image / 255.0
            image = rearrange(image, 'h w c -> 1 c h w')
            depth = self.model.infer(image)
            depth = depth[0, 0].cpu().numpy()

        vmin = np.percentile(depth, 2)
        vmax = np.percentile(depth, 85)

        depth -= vmin
        depth /= vmax - vmin
        depth = 1.0 - depth
        depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)
        return depth_image


#############################################
#                   DSINE                   #
#   https://github.com/baegwangbin/DSINE    #
#############################################
from .models.dsine import DSINE, utils as dsine_utils, resize_image_with_pad

class DsineEstimate:

    model_name = "dsine"

    def __init__(self, device = DEVICE, preload: bool = False):
        self.device = device
        self.model = None
        if preload:
            self.load_model()

    def load_model(self):
        local_model_path = CTRL_ANNOT_LOCAL_MODELS[self.model_name]
        if not os.path.exists(local_model_path):
            remote_model_path = CTRL_ANNOT_REMOTE_MODELS[self.model_name]
            load_file_from_url(remote_url=remote_model_path, 
                                local_file=local_model_path)
        model = DSINE()
        model.pixel_coords = model.pixel_coords.to(self.device)
        model = dsine_utils.load_checkpoint(modelpath, model)
        model.eval()
        self.model = model.to(self.device)
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def offload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image, new_fov=60.0, iterations=5, resolution=512):
        if self.model is None:
            self.load_model()
        self.model.num_iter = iterations

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        assert input_image.ndim == 3
        
        orig_H, orig_W = input_image.shape[:2]
        l, r, t, b = dsine_utils.pad_input(orig_H, orig_W)
        input_image, \
        remove_pad = resize_image_with_pad(input_image, resolution)

        with torch.no_grad():
            image = torch.from_numpy(input_image).float().to(self.device)
            image = image / 255.0
            image = rearrange(image, 'h w c -> 1 c h w')
            image = self.transform(image)
            
            intrins = dsine_utils.get_intrins_from_fov(new_fov=new_fov, H=orig_H, W=orig_W, device=self.device).unsqueeze(0)
            intrins[:, 0, 2] += l
            intrins[:, 1, 2] += t
            
            normal = self.model(image, intrins=intrins)[-1]
            normal = normal[:, :, t:t+orig_H, l:l+orig_W]
            normal = ((normal + 1) * 0.5).clip(0, 1)
            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()

        normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)
        return remove_pad(normal_image)


#############################
#       Universal           #
#############################
all_options_depth = [
    "leres", "leres++",
    "dpt_large", "dpt_hybrid", "midas_v21", "midas_v21s",
    "normalbae", "dsine",
    "zoedepth"
]

def apply_depth(input_image, model: str = "normal", *, **kwargs):

    if model == "leres":
        estimator = LeResEstimate(boost=False)
        return estimator(input_image, **kwargs)

    elif model in ["leres++", "leres_boost"]:
        estimator = LeResEstimate(boost=True)
        return estimator(input_image, **kwargs)

    elif model in ["dpt_large", "dpt_hybrid", "midas_v21", "midas_v21s"]:
        estimator = MiDaSEstimate(model)
        return estimator(input_image, **kwargs)

    elif model in ["zoe", "zoedepth"]:
        estimator = ZoeDepthEstimate()
        return estimator(input_image)

    elif model == "normalbae":
        estimator = NormalBaeEstimate()
        return estimator(input_image)

    elif model in ["normal_dsine", "dsine"]:
        estimator = DsineEstimate()
        return estimator(input_image, **kwargs)

    else:
        raise ValueError(f"model = {model} is not supported!")


