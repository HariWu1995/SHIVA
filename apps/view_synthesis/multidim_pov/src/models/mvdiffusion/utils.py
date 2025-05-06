import cv2
import numpy as np
from exiftool import ExifToolHelper


def preprocess_image(img, resolution: int):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_and_center_crop(img, resolution)
    img = img / 127.5 - 1
    return img


def multiview_Rs_Ks(resolution: int, num_views: int = 8):
    Rs = []
    Ks = []
    angle = 360 // num_views
    for i in range(num_views):
        degree = (angle * i) % 360
        K, R = get_K_R(90, degree, 0, resolution, resolution)
        Rs.append(R)
        Ks.append(K)
    return Rs, Ks


# read prompts from image
def read_prompts_from_image(image_path):
    with ExifToolHelper() as reader:
        exif_info = reader.get_metadata(image_path)
        print(f'Metadata: {exif_info}')        

        if 'PNG:Parameters' in exif_info[0]:
            parameters = exif_info[0]['PNG:Parameters']
            parsed_parameters = parameters.split("\n")        
            positive_prompt = parsed_parameters[0]
            negative_prompt = parsed_parameters[1]
 
        else:
            positive_prompt = None
            negative_prompt = None

    return positive_prompt, negative_prompt


def get_K_R(FOV, THETA, PHI, height, width):
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = ( width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)

    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R


def resize_and_center_crop(img, size):
    H, W, _ = img.shape
    if H == W:
        img = cv2.resize(img, (size, size))
    elif H > W:
        current_size = int(size * H / W)
        img = cv2.resize(img, (size, current_size))
        margin_l = (current_size - size) // 2
        margin_r =  current_size - size - margin_l
        img = img[margin_l:-margin_r, :]
    else:
        current_size = int(size * W / H)
        img = cv2.resize(img, (current_size, size))
        margin_t = (current_size - size) // 2
        margin_b =  current_size - size - margin_t
        img = img[:, margin_t:-margin_b]
    return img

