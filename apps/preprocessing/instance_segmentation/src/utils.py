import numpy as np
from PIL import Image, ImageDraw


def plot_boxes(img, boxes):
	img_pil = Image.fromarray(np.uint8(img * 255)).convert('RGB')
	drawer = ImageDraw.Draw(img_pil)
	for box in boxes:
		color = tuple(np.random.randint(0, 255, size=3).tolist())
		x0, y0, x1, y1 = box
		x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
		drawer.rectangle([x0, y0, x1, y1], outline=color, width=6)
	return img_pil

