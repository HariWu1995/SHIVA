import cv2


def resize_shortest_edge(image, size):
    h, w = image.shape[:2]
    if h < w:
        new_h = size
        new_w = int(round(w / h * size))
    else:
        new_w = size
        new_h = int(round(h / w * size))
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image


def apply_binary(img, threshold: int | float = 128):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if threshold < 0:
        otsu_threshold, \
            img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, img_bin = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)


def apply_color(img, res: int = 512):
    img = resize_shortest_edge(img, res)
    Hrz, Wrz = img.shape[:2]
    Hcl = Hrz // 64
    Wcl = Wrz // 64
    out = cv2.resize(img, (Wcl, Hcl), interpolation=cv2.INTER_CUBIC)  
    out = cv2.resize(out, (Wrz, Hrz), interpolation=cv2.INTER_NEAREST)
    return out


