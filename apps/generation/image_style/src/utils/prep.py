from PIL import Image
import numpy as np


def resize_width_height(width, height, min_short_side: int = 512, 
                                        max_long_side: int = 1024):
    if width < height:
        if width < min_short_side:
            scale_factor = min_short_side / width
            new_width = min_short_side
            new_height = int(height * scale_factor)
        else:
            new_width, new_height = width, height
    else:
        if height < min_short_side:
            scale_factor = min_short_side / height
            new_width = int(width * scale_factor)
            new_height = min_short_side
        else:
            new_width, new_height = width, height

    if max(new_width, new_height) > max_long_side:
        scale_factor = max_long_side / max(new_width, new_height)
        new_width  = int(new_width * scale_factor)
        new_height = int(new_height * scale_factor)

    return new_width, new_height


def resize_image(image, max_long_side: int = 1024, 
                        min_short_side: int = 1024):

    new_width, new_height = resize_width_height(image.size[0], image.size[1],
                                                min_short_side=min_short_side, 
                                                max_long_side=max_long_side)
    height = new_height // 16 * 16
    width  =  new_width // 16 * 16
    image = image.resize((width, height))

    return width, height, image

