import os
import functools

import torch
from einops import rearrange

from PIL import Image
import cv2

import numpy as np
import random as rd

from .utils import DEVICE, CTRL_ANNOT_LOCAL_MODELS, CTRL_ANNOT_REMOTE_MODELS, load_file_from_url


#####################################################
#       Hollistically-Nested Edge Detection         #
#####################################################
from .models.hed import ControlNetHED

class HolyEdgeDetection:

    model_name = "hed"

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
        net = ControlNetHED()
        ckpt = torch.load(local_model_path, map_location=self.device)
        net.load_state_dict(ckpt)
        net.eval()
        self.model = net.to(self.device).float()

    def offload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, img: np.ndarray | Image.Image, is_safe: bool = False):

        if self.model is None:
            self.load_model()

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        assert img.ndim == 3
        H, W, C = img.shape

        with torch.no_grad():
            image_hed = torch.from_numpy(img.copy()).float().to(self.device)
            image_hed = rearrange(image_hed, 'h w c -> 1 c h w')

            edges = self.model(image_hed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))

        if is_safe:
            edge = safe_step(edge)
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
        return edge


#########################################
#           Canny - Traditional         #
#########################################
class CannyEdgeDetection:

    def auto_threshold(
        self, 
        low_threshold: int = None,
        high_threshold: int = None, 
            is_noisy: bool = False,
            contrast: str = "normal",
            structure: str = "normal",      # visual quality and structure of edges
    ):
        if low_threshold is None and high_threshold is not None:
            low_threshold = max(0, high_threshold // 3)
        elif high_threshold is None and low_threshold is not None:
            high_threshold = min(255, low_threshold * 3)
        elif high_threshold is None and low_threshold is None:
            if is_noisy:
                low_threshold = 85
                high_threshold = 225
            elif contrast == "high" and structure == "clean":
                low_threshold = 75
                high_threshold = 175
            elif contrast == "low" and structure == "smooth":
                low_threshold = 35
                high_threshold = 115
            else:
                low_threshold = 35
                high_threshold = 115
        return low_threshold, high_threshold

    def __call__(
        self, 
        img, 
        low_threshold: int = None,
        high_threshold: int = None, 
            is_noisy: bool = False,
            structure: str = "normal",
            contrast: str = "normal",
    ):
        low_threshold, \
        high_threshold = self.auto_threshold(low_threshold, 
                                            high_threshold, structure, contrast, is_noisy)
        return cv2.Canny(img, low_threshold, high_threshold)


#################################
#           LineArt             #
#################################
from .models.lineart import Generator as LineArtGenerator

class LineArtDetection:

    model_default = 'lineart'
    model_coarse = 'linecoarse'

    def __init__(self, device = DEVICE, coarse: bool = False, preload: bool = False):
        self.device = device
        self.coarse = coarse
        self.model = None
        if preload:
            self.load_model()

    def load_model(self):
        model_name = self.model_coarse if self.coarse else \
                     self.model_default
        local_model_path = CTRL_ANNOT_LOCAL_MODELS[model_name]
        if not os.path.exists(local_model_path):
            remote_model_path = CTRL_ANNOT_REMOTE_MODELS[model_name]
            load_file_from_url(remote_url=remote_model_path, 
                                local_file=local_model_path)
        model = LineArtGenerator(3, 1, 3)
        model.load_state_dict(torch.load(local_model_path, map_location='cpu'))
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
            line = self.model(image)[0][0]

        line = line.cpu().numpy()
        line = (line * 255.0).clip(0, 255).astype(np.uint8)
        return line


#####################################
#           LineArt - Anime         #
#####################################
from .models.lineart_anime import UnetGenerator as LineAnimeGenerator

class LineAnimeDetection:

    model_name = 'lineanime'

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

        norm = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        net = LineAnimeGenerator(3, 1, 8, 64, norm_layer=norm, use_dropout=False)
        ckpt = torch.load(local_model_path)
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

    def __call__(self, input_image):
        if self.model is None:
            self.load_model()

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        assert input_image.ndim == 3
        H, W, C = input_image.shape

        Hn = 256 * int(np.ceil(float(H) / 256.0))
        Wn = 256 * int(np.ceil(float(W) / 256.0))

        img = cv2.resize(input_image, (Wn, Hn), interpolation=cv2.INTER_CUBIC)

        with torch.no_grad():
            image = torch.from_numpy(img).float().to(self.device)
            image = image / 127.5 - 1.0
            image = rearrange(image, 'h w c -> 1 c h w')

            line = self.model(image)[0, 0] * 127.5 + 127.5
            line = line.cpu().numpy()

        line = cv2.resize(line, (W, H), interpolation=cv2.INTER_CUBIC)
        line = line.clip(0, 255).astype(np.uint8)
        return line


#####################################
#           LineArt - Manga         #
#####################################
from .models.lineart_manga import res_skip as LineMangaGenerator

class LineMangaDetection:

    model_name = 'linemanga'

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

        net = LineMangaGenerator()
        ckpt = torch.load(local_model_path)
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

    def __call__(self, input_image):
        if self.model is None:
            self.load_model()

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        assert input_image.ndim == 3

        img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        img = np.ascontiguousarray(img.copy()).copy()
        with torch.no_grad():
            image = torch.from_numpy(img).float().to(self.device)
            image = rearrange(image, 'h w -> 1 1 h w')
            line = self.model(image)

        line = 255 - line.cpu().numpy()[0, 0]
        line = line.clip(0, 255).astype(np.uint8)
        return line


#############################################
#               Mobile-LSD                  #
#   https://github.com/navervision/mlsd     #
#############################################
from .models.mlsd import MobileV2_MLSD_Tiny, MobileV2_MLSD_Large, mlsd_prediction

class MobileLSDetection:

    model_name = 'mlsd'

    def __init__(self, device = DEVICE, variant: str = "tiny", preload: bool = False):
        self.device = device
        assert variant in ['large','tiny']
        self.variant = variant
        self.model = None
        if preload:
            self.load_model()

    def load_model(self):
        model_name = f"{self.model_name}_{self.variant}"
        local_model_path = CTRL_ANNOT_LOCAL_MODELS[model_name]
        if not os.path.exists(local_model_path):
            remote_model_path = CTRL_ANNOT_REMOTE_MODELS[model_name]
            load_file_from_url(remote_url=remote_model_path, 
                                local_file=local_model_path)

        model = MobileV2_MLSD_Large() if self.variant == "large" else \
                MobileV2_MLSD_Tiny()
        model.load_state_dict(torch.load(local_model_path), strict=True)
        model.eval()
        self.model = model.to(device=self.device)

    def offload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image, thresh_v, thresh_d):
        if self.model is None:
            self.load_model()

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        assert input_image.ndim == 3
        H, W  = input_image.shape[:2]

        img = input_image
        img_output = np.zeros_like(img)

        with torch.no_grad():
            lines = mlsd_prediction(img, self.model, [H, W], 
                                    score_thresh=thresh_v, 
                                    dist_thresh=thresh_d, 
                                    device=self.device)
        for line in lines:
            x_start, y_start, x_end, y_end = [int(val) for val in line]
            cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        return img_output[:, :, 0]


#########################################
#   PiDiNet - Pixel Difference Network  #
# https://github.com/hellozhuo/pidinet  #
#########################################
from .models.pidinet import load_pidinet, safe_step

class PiDiNetDetection:

    model_name = 'pidinet'

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

        net = load_pidinet()
        ckpt = torch.load(local_model_path)
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

    def __call__(self, input_image, is_safe=False, apply_fliter=False):
        if self.model is None:
            self.load_model()

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        assert input_image.ndim == 3
        image = input_image[:, :, ::-1].copy()

        with torch.no_grad():
            image = torch.from_numpy(image).float().to(device=self.device)
            image = image / 255.0
            image = rearrange(image, 'h w c -> 1 c h w')
            edge = self.model(image)[-1]
            edge = edge.cpu().numpy()

        if apply_fliter:
            edge = edge > 0.5 
        if is_safe:
            edge = safe_step(edge)
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
        return edge[0][0] 


#############################################################
#       Tiny & Efficient Model for the Edge Detection       #
#               https://github.com/xavysp/TEED              #
#############################################################
from .models.teed import TED

class TEEDetection:

    def __init__(self, device = DEVICE, preload: bool = False, use_mteed: bool = False):
        self.device = device
        self.model_name = "mteed" if use_mteed else "teed"
        self.model = None
        if preload:
            self.load_model()

    def load_model(self):
        local_model_path = CTRL_ANNOT_LOCAL_MODELS[self.model_name]
        if not os.path.exists(local_model_path):
            remote_model_path = CTRL_ANNOT_REMOTE_MODELS[self.model_name]
            load_file_from_url(remote_url=remote_model_path, 
                                local_file=local_model_path)
        model = TED()
        ckpt = torch.load(local_model_path)
        for key in list(ckpt.keys()):
            if 'module.' in key:
                ckpt[key.replace('module.', '')] = ckpt[key]
                del ckpt[key]
        model.load_state_dict(ckpt)
        model.eval()
        self.model = model.to(self.device)

    def offload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, image: np.ndarray, safe_steps: int = 2) -> np.ndarray:

        if self.model is None:
            self.load_model()

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        assert input_image.ndim == 3
        H, W = input_image.shape[:2]

        with torch.no_grad():
            image = torch.from_numpy(input_image.copy()).float().to(self.device)
            image = rearrange(image, "h w c -> 1 c h w")
            edges = self.model(image)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]

        edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
        edges = np.stack(edges, axis=2)
        edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
        if safe_steps != 0:
            edge = safe_step(edge, safe_steps)
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
        return edge


#############################
#       Universal           #
#############################
all_options_edge = [
    "canny", 
    "hed", 
    "pidinet",
    "lineart", "lineart-coarse", "lineart-anime",
    "mlsd_large", "mlsd_tiny",
    "teed", "mteed",
]

def apply_edge(input_image, model: str = "canny", *, **kwargs):

    if model == "canny":
        detector = CannyEdgeDetection()
        return detector(input_image, **kwargs)

    elif model == "hed":
        detector = HolyEdgeDetection()
        return detector(input_image, **kwargs)

    elif model == "pidinet":
        detector = PiDiNetDetection()
        return detector(input_image, **kwargs)

    elif model in ["mlsd_large", "mlsd_tiny"]:
        detector = MobileLSDetection(variant=model.split('_'[1]))
        return detector(input_image, **kwargs)

    elif model in ["teed", "mteed"]:
        detector = TEEDetection(use_mteed = model == "mteed")
        return detector(input_image, **kwargs)

    elif model == "lineart":
        detector = LineArtDetection(coarse=False)
        return detector(input_image)

    elif model == "lineart-coarse":
        detector = LineArtDetection(coarse=True)
        return detector(input_image)

    elif model == "lineart-anime":
        detector = LineAnimeDetection()
        return detector(input_image)

    else:
        raise ValueError(f"model = {model} is not supported!")

