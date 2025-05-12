from PIL import Image
from typing import List, Literal, Optional, Tuple, Union
from colorama import Fore, Style, init
init(autoreset=True)

import math
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF


def get_resizing_factor(
    target_shape: Tuple[int, int],  # H, W
    current_shape: Tuple[int, int],  # H, W
    cover_target: bool = True,
) -> float:
    r_bound  =  target_shape[1] /  target_shape[0]
    aspect_r = current_shape[1] / current_shape[0]
    if r_bound >= 1.0:
        if cover_target:
            if aspect_r >= r_bound:
                factor = min(target_shape) / min(current_shape)
            elif aspect_r < 1.0:
                factor = max(target_shape) / min(current_shape)
            else:
                factor = max(target_shape) / max(current_shape)
        else:
            if aspect_r >= r_bound:
                factor = max(target_shape) / max(current_shape)
            elif aspect_r < 1.0:
                factor = min(target_shape) / max(current_shape)
            else:
                factor = min(target_shape) / min(current_shape)
    else:
        if cover_target:
            if aspect_r <= r_bound:
                factor = min(target_shape) / min(current_shape)
            elif aspect_r > 1.0:
                factor = max(target_shape) / min(current_shape)
            else:
                factor = max(target_shape) / max(current_shape)
        else:
            if aspect_r <= r_bound:
                factor = max(target_shape) / max(current_shape)
            elif aspect_r > 1.0:
                factor = min(target_shape) / max(current_shape)
            else:
                factor = min(target_shape) / min(current_shape)
    return factor


def get_wh_with_fixed_shortest_side(w, h, size):
    # size is smaller or equal to zero, we return original w h
    if size is None or size <= 0:
        return w, h
    if w < h:
        new_w = size
        new_h = int(size * h / w)
    else:
        new_h = size
        new_w = int(size * w / h)
    return new_w, new_h


def load_img_and_K(
    image_path_or_size: Union[str, torch.Size],
    size: Optional[Union[int, Tuple[int, int]]],
    scale: float = 1.0,
    center: Tuple[float, float] = (0.5, 0.5),
    K: torch.Tensor | None = None,
    size_stride: int = 1,
    center_crop: bool = False,
    image_as_tensor: bool = True,
    context_rgb: np.ndarray | None = None,
    device: str = "cuda",
):
    if isinstance(image_path_or_size, torch.Size):
        image = Image.new("RGBA", image_path_or_size[::-1])
    else:
        image = Image.open(image_path_or_size).convert("RGBA")

    w, h = image.size
    if size is None:
        size = (w, h)

    image = np.array(image).astype(np.float32) / 255
    if image.shape[-1] == 4:
        rgb   = image[:, :, :3]
        alpha = image[:, :, 3:]
        if context_rgb is not None:
            image = rgb * alpha + (1 - alpha) * context_rgb
        else:
            image = rgb * alpha + (1 - alpha)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).to(dtype=torch.float32)
    image = image.unsqueeze(0)

    if isinstance(size, (tuple, list)):
        # => if size is a tuple or list, we first rescale to fully cover the `size`
        # area and then crop the `size` area from the rescale image
        W, H = size
    else:
        # => if size is int, we rescale the image to fit the shortest side to size
        # => if size is None, no rescaling is applied
        W, H = get_wh_with_fixed_shortest_side(w, h, size)

    W = math.floor(W / size_stride + 0.5) * size_stride
    H = math.floor(H / size_stride + 0.5) * size_stride

    rfs = get_resizing_factor((math.floor(H * scale), 
                               math.floor(W * scale)), (h, w))
    re_size = rh, rw = [int(np.ceil(rfs * s)) for s in (h, w)]
    image = F.interpolate(image, re_size, mode="area", antialias=False)

    if scale < 1.0:
        pw = math.ceil((W - re_size[1]) * 0.5)
        ph = math.ceil((H - re_size[0]) * 0.5)
        image = F.pad(image, (pw, pw, ph, ph), "constant", 1.0)

    cy_center = int(center[1] * image.shape[-2])
    cx_center = int(center[0] * image.shape[-1])
    if center_crop:
        side = min(H, W)
        ct = max(0, cy_center - side // 2)
        cl = max(0, cx_center - side // 2)
        ct = min(ct, image.shape[-2] - side)
        cl = min(cl, image.shape[-1] - side)
        image = tvF.crop(image, top=ct, left=cl, height=side, width=side)
    else:
        ct = max(0, cy_center - H // 2)
        cl = max(0, cx_center - W // 2)
        ct = min(ct, image.shape[-2] - H)
        cl = min(cl, image.shape[-1] - W)
        image = tvF.crop(image, top=ct, left=cl, height=H, width=W)

    if K is not None:
        K = K.clone()
        if  torch.all(K[:2, -1] >= 0) \
        and torch.all(K[:2, -1] <= 1):
            K[:2] *= K.new_tensor([rw, rh])[:, None]            # normalized K
        else:
            K[:2] *= K.new_tensor([rw / w, rh / h])[:, None]    # unnormalized K
        K[:2, 2] -= K.new_tensor([cl, ct])

    if image_as_tensor:
        # tensor of shape (1, 3, H, W) with values ranging from (-1, 1)
        image = image.to(device) * 2.0 - 1.0
    else:
        # PIL Image with values ranging from (0, 255)
        image = image.permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).astype(np.uint8))
    return image, K


def transform_img_and_K(
    image: torch.Tensor,
    size: Union[int, Tuple[int, int]],
    scale: float = 1.0,
    center: Tuple[float, float] = (0.5, 0.5),
    K: torch.Tensor | None = None,
    size_stride: int = 1,
    mode: str = "crop",
):
    assert mode in ["crop","pad","stretch"], \
        f"`mode` should be one of ['crop', 'pad', 'stretch'], got {mode}"

    h, w = image.shape[-2:]
    if isinstance(size, (tuple, list)):
        # => if size is a tuple or list, we first rescale to fully cover the `size`
        # area and then crop the `size` area from the rescale image
        W, H = size
    else:
        # => if size is int, we rescale the image to fit the shortest side to size
        # => if size is None, no rescaling is applied
        W, H = get_wh_with_fixed_shortest_side(w, h, size)

    W = math.floor(W / size_stride + 0.5) * size_stride
    H = math.floor(H / size_stride + 0.5) * size_stride

    if mode == "stretch":
        rh, rw = H, W
    else:
        rfs = get_resizing_factor((H, W), (h, w), cover_target = mode != "pad")
        (rh, rw) = [int(np.ceil(rfs * s)) for s in (h, w)]

    rh, rw = int(rh / scale), int(rw / scale)
    image = F.interpolate(image, (rh, rw), mode="area", antialias=False)

    cy_center = int(center[1] * image.shape[-2])
    cx_center = int(center[0] * image.shape[-1])

    if mode != "pad":
        ct = max(0, cy_center - H // 2)
        cl = max(0, cx_center - W // 2)
        ct = min(ct, image.shape[-2] - H)
        cl = min(cl, image.shape[-1] - W)
        image = tvF.crop(image, top=ct, left=cl, height=H, width=W)
        pl, pt = 0, 0
    else:
        pt = max(0, H // 2 - cy_center)
        pl = max(0, W // 2 - cx_center)
        pb = max(0, H - pt - image.shape[-2])
        pr = max(0, W - pl - image.shape[-1])
        image = tvF.pad(image, [pl, pt, pr, pb])
        cl, ct = 0, 0

    if K is not None:
        K = K.clone()
        # K[:, :2, 2] += K.new_tensor([pl, pt])
        if  torch.all(K[:, :2, -1] >= 0) \
        and torch.all(K[:, :2, -1] <= 1):
            K[:, :2] *= K.new_tensor([rw, rh])[None, :, None]           # normalized K
        else:
            K[:, :2] *= K.new_tensor([rw / w, rh / h])[None, :, None]   # unnormalized K
        K[:, :2, 2] += K.new_tensor([pl - cl, pt - ct])

    return image, K

