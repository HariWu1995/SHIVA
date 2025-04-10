from .utils import remove_background
from .models import sessions_names as models
from .session import new_session

default_model = 'u2net'
all_models = [
    "u2net","u2netp","u2net_human","u2net_cloth",
    "silueta",
    "isnet-general","isnet-anime",
]


def run_rembg(img, alpha, alpha_fg_threshold, alpha_bg_threshold, alpha_erosion_size, model):
    
    masked_obj = remove_background(img, alpha, 
                                        alpha_fg_threshold, 
                                        alpha_bg_threshold, 
                                        alpha_erosion_size, model)
    # Split layers
    masked_obj = masked_obj.convert("RGBA")
    *rgb, alpha = masked_obj.split()

    mask = alpha.point(lambda p: 255 - p)

    # obj = Image.merge("RGB", rgb)
    # obj.putalpha(mask.convert('L'))

    return img, mask
