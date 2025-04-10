import os
import gc

from PIL import Image, ImageDraw
import cv2

import numpy as np
import torch

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator as SamMaskGenerator
from transformers import OwlViTProcessor, OwlViTForObjectDetection as OwlViTDetector
from .utils import plot_boxes


CHECKPOINT_ROOT = os.environ.get('SHIVA_CKPT_ROOT', None)
if CHECKPOINT_ROOT is not None:
    SAM_DIR = Path(CHECKPOINT_ROOT) / 'sam'
	OWL_DIR = Path(CHECKPOINT_ROOT) / 'owlvit-base-patch32'
else:
    SAM_DIR = Path(__file__).parents[5] / 'checkpoints/sam'
    OWL_DIR = Path(__file__).parents[5] / 'checkpoints/owlvit-base-patch32'

OWL_DIR = str(OWL_DIR)
sam_models = {
	'vit_b': str(OWL_DIR / 'sam_vit_b_01ec64.pth'),
	'vit_l': str(OWL_DIR / 'sam_vit_l_0b3195.pth'),
	'vit_h': str(OWL_DIR / 'sam_vit_h_4b8939.pth'),
}


def segment_one(img, mask_generator, seed=None):
	if seed is not None:
		np.random.seed(seed)
	masks = mask_generator.generate(img)
	sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
	mask_all = np.ones((img.shape[0], img.shape[1], 3))
	for ann in sorted_anns:
		m = ann['segmentation']
		color_mask = np.random.random((1, 3)).tolist()[0]
		for i in range(3):
			mask_all[m == True, i] = color_mask[i]
	result = img / 255 * 0.3 + mask_all * 0.7
	return result, mask_all


def generator_inference(
		device, model_type, input_x, points_per_side, 
		pred_iou_thresh, stability_score_thresh,
		min_mask_region_area, stability_score_offset, 
		box_nms_thresh, crop_n_layers, crop_nms_thresh,
		progress=None,
	):
	# SAM
	segmentor = sam_model_registry[model_type](checkpoint=sam_models[model_type]).to(device)
	mask_generator = SamMaskGenerator(
		segmentor,
		point_grids=None,
		points_per_side=points_per_side,
		pred_iou_thresh=pred_iou_thresh,
		stability_score_thresh=stability_score_thresh,
		stability_score_offset=stability_score_offset,
		 box_nms_thresh=box_nms_thresh,
		crop_nms_thresh=crop_nms_thresh,
		crop_n_layers=crop_n_layers,
		crop_overlap_ratio=512 / 1500,
		crop_n_points_downscale_factor=1,
		min_mask_region_area=min_mask_region_area,
		output_mode='binary_mask',
	)

	# input is image, type: numpy
	if type(input_x) == np.ndarray:
		result, mask_all = segment_one(input_x, mask_generator)
		return result, mask_all

	# input is video, type: path (str)
	elif isinstance(input_x, str):
		cap = cv2.VideoCapture(input_x)  # read video
		frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = int(cap.get(cv2.CAP_PROP_FPS))

		out_path = "temp/sam/output.mp4"
		fourcc = cv2.VideoWriter_fourcc(list('x264'))
		writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H), isColor=True)
		while True:
			ret, frame = cap.read()  # read a frame
			if ret:
				result, mask_all = segment_one(frame, mask_generator, seed=2023)
				result = (result * 255).astype(np.uint8)
				writer.write(result)
			else:
				break
		writer.release()
		cap.release()
		return out_path


def predictor_inference(device, model_type, input_x, input_text, selected_points, owl_vit_threshold=0.1):
	# SAM
	segmentor = sam_model_registry[model_type](checkpoint=models[model_type]).to(device)
	predictor = SamPredictor(segmentor)
	predictor.set_image(input_x)  # Process the image to produce an image embedding

	if input_text != '':
		# split input text
		input_text = [input_text.split(',')]

		# OWL-ViT model
		if not os.path.isdir(OWL_DIR):
			OWL_DIR = "google/owlvit-base-patch32"
		processor = OwlViTProcessor.from_pretrained(OWL_DIR)
		detector = OwlViTDetector.from_pretrained(OWL_DIR).to(device)

		# get outputs
		target_size = torch.Tensor([input_x.shape[:2]]).to(device)
		input_text = processor(text=input_text, images=input_x, return_tensors="pt").to(device)
		outputs = detector(**input_text)
		results = processor.post_process_object_detection(outputs=outputs, 
													 target_sizes=target_size,
		                                                threshold=owl_vit_threshold)
		# get the box with best score
		scores = torch.sigmoid(outputs.logits)
		# best_scores, best_idxs = torch.topk(scores, k=1, dim=1)
		# best_idxs = best_idxs.squeeze(1).tolist()

		i = 0  # Retrieve predictions for the first image for the corresponding text queries
		boxes_tensor = results[i]["boxes"]  # [best_idxs]
		boxes = boxes_tensor.cpu().detach().numpy()
		# boxes = boxes[np.newaxis, :, :]
		transformed_boxes = predictor.transform.apply_boxes_torch(torch.Tensor(boxes).to(device),
		                                                          input_x.shape[:2])  # apply transform to original boxes
		# transformed_boxes = transformed_boxes.unsqueeze(0)
		print(transformed_boxes.size(), boxes.shape)

	else:
		transformed_boxes = None

	# points
	if len(selected_points) != 0:
		points = torch.Tensor([p for p, _ in selected_points]).to(device).unsqueeze(1)
		labels = torch.Tensor([int(l) for _, l in selected_points]).to(device).unsqueeze(1)
		transformed_points = predictor.transform.apply_coords_torch(points, input_x.shape[:2])
		print(points.size(), transformed_points.size(), labels.size(), input_x.shape, points)
	else:
		transformed_points, labels = None, None

	# predict segmentation according to the boxes
	masks, scores, logits = predictor.predict_torch(
		point_coords=transformed_points,
		point_labels=labels,
		boxes=transformed_boxes,  # only one box
		multimask_output=False,
	)
	masks = masks.cpu().detach().numpy()
	mask_all = np.ones((input_x.shape[0], input_x.shape[1], 3))
	for ann in masks:
		color_mask = np.random.random((1, 3)).tolist()[0]
		for i in range(3):
			mask_all[ann[0] == True, i] = color_mask[i]
	img = input_x / 255 * 0.3 + mask_all * 0.7
	if input_text != '':
		img = plot_boxes(img, boxes_tensor)  # image + mask + boxes

	# free the memory
	if input_text != '':
		owlvit_model.cpu()
		del owlvit_model
	del input_text
	gc.collect()
	torch.cuda.empty_cache()

	return img, mask_all


def run_inference(device, model_type, input_x, input_text, 
				  points_per_side = 32, 
				  selected_points = [], 
			  min_mask_region_area = 0, 
			stability_score_offset = 1, 
			stability_score_thresh = 0.95, 
				   pred_iou_thresh = 0.75, 
			  		box_nms_thresh = 0.75, 
				    owl_vit_thresh = 0.11,
				   crop_nms_thresh = 0.75, 
				    crop_n_layers = 0, 
						progress = None):
	
	# if input_x is int, the image is selected from examples
	# if isinstance(input_x, int):
	# 	input_x = cv2.imread(image_examples[input_x][0])
	# 	input_x = cv2.cvtColor(input_x, cv2.COLOR_BGR2RGB)

	# user input text or points
	if (input_text != '' and not isinstance(input_x, str)) or len(selected_points) != 0:
		print('\nRun predictor_inference')
		print('\tprompt text: ', input_text)
		print('\tprompt points length: ', len(selected_points))
		return predictor_inference(device, model_type, input_x, input_text, 
								   selected_points, owl_vit_threshold)

	else:
		print('\nRun generator_inference')
		return generator_inference(device, model_type, input_x, points_per_side, 
								   pred_iou_thresh, stability_score_thresh,
								   min_mask_region_area, stability_score_offset, 
								   box_nms_thresh, crop_n_layers, crop_nms_thresh, progress)

