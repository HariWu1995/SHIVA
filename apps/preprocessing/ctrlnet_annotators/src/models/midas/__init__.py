from .dpt_depth import DPTDepthModel
from .midas_net import MidasNet
from .midas_net_custom import MidasNet_small
from .transforms import Resize, NormalizeImage, PrepareForNet

from torchvision.transforms import Compose


midas_backbones = dict(
    dpt_large = "vitl16_384",
    dpt_hybrid = "vitb_rn50_384",
    midas_v21s = "efficientnet_lite3",
)


def midas_transform(model_type):
    # https://github.com/isl-org/MiDaS/blob/master/run.py
    # load transform only
    if model_type in ["dpt_large", "dpt_hybrid"]:
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        Normalize = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type in ["midas_v21", "midas_v21s"]:
        if model_type == "midas_v21":
            net_w, net_h = 384, 384
        elif model_type == "midas_v21s":
            net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        Normalize = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    else:
        assert False, f"model_type '{model_type}' not implemented, use: --model_type large"

    transform = Compose([
        Resize(net_w, net_h, resize_target=None, resize_method=resize_mode,
                            ensure_multiple_of=32, keep_aspect_ratio=True,
                            image_interpolation_method=cv2.INTER_CUBIC),
        Normalize,
        PrepareForNet(),
    ])

    return transform
