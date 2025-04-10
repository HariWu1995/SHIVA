import numpy as np
import cv2

from PIL import Image, ImageOps


def save_all(image, mask):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    image.save('./temp/rembg/image.png')
    mask.save('./temp/rembg/mask.png')


def invert_mask(mask):
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    mask = ImageOps.invert(mask)
    return mask


def apply_mask(image, mask):
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask).convert('L')
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    image.putalpha(mask)
    return image


def expand_img(image, left: int = 0, right: int = 0, 
                       top: int = 0, bottom: int = 0):
    paddings = (top, bottom, left, right)
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = cv2.copyMakeBorder(image, *paddings, cv2.BORDER_REPLICATE)
    return image

