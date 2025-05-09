import os
import cv2
from PIL import Image

import numpy as np
import torch
import torchvision as tv
from einops import rearrange

from .utils import DEVICE, CTRL_ANNOT_LOCAL_MODELS, CTRL_ANNOT_REMOTE_MODELS, load_file_from_url


#############################
#       DensePose           #
#############################
from .models.densepose import (
    densepose_chart_predictor_output_to_result_with_confidences as densepose_postprocess, 
    visualize as densepose_visualize,
)

class DensePoseEstimate:

    model_name = "densepose"
    N_PART_LABELS = 24

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
        model = torch.jit.load(local_model_path, map_location="cpu")
        model.eval()
        self.model = model.to(device=self.device)

    def offload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image: np.ndarray | Image.Image, cmap: str = "viridis"):

        if self.model is None:
            self.load_model()

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        H, W  = input_image.shape[:2]

        hint_canvas = np.zeros([H, W], dtype=np.uint8)
        hint_canvas = np.tile(hint_canvas[:, :, np.newaxis], [1, 1, 3])

        input_image = rearrange(
            torch.from_numpy(input_image).to(device=self.device), 'h w c -> c h w')
        pred_boxes, corase_segm, refine_segm, u, v = self.model(input_image)

        densepose_results = [
            densepose_postprocess(
                 pred_boxes[i:i+1], 
                corase_segm[i:i+1], 
                refine_segm[i:i+1], 
                          u[i:i+1], 
                          v[i:i+1]) for i in range(len(pred_boxes))
        ]

        hint_image = densepose_visualize(hint_canvas, densepose_results, 
                                        num_parts=self.N_PART_LABELS, cmap=cmap)
        return hint_image


#############################
#           MMPose          #
#############################
try:
    import mmcv
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_top_down_pose_model
    from mmpose.apis import init_pose_model, process_mmdet_results, vis_pose_result
    IS_MMPOSE_INSTALLED = True
except (ModuleNotFoundError, ImportError) as e:
    IS_MMPOSE_INSTALLED = False

from .models.mmpose import *

class MMPoseEstimate:

    model_name = "mmpose"
    config_file = dict(det=det_config_file, kpt=kpt_config_file)
    person_id = det_cat_id
    bbox_thresh = bbox_thresh
    skeleton = skeleton
    pose_link_color = pose_link_color
    pose_kpt_color = pose_kpt_color

    def __init__(self, device = DEVICE, preload: bool = False):
        self.device = device
        self.det_model = None
        self.kpt_model = None
        if preload:
            self.load_model()

    def load_model(self):
        for mtype in ['det', 'kpt']:
            model_name = f'{self.model_name}_{mtype}'
            local_model_path = CTRL_ANNOT_LOCAL_MODELS[model_name]
            if not os.path.exists(local_model_path):
                remote_model_path = CTRL_ANNOT_REMOTE_MODELS[model_name]
                load_file_from_url(remote_url=remote_model_path, 
                                    local_file=local_model_path)

        det_config_mmcv = mmcv.Config.fromfile(self.config_file['det'])
        kpt_config_mmcv = mmcv.Config.fromfile(self.config_file['kpt'])

        det_model_local = CTRL_ANNOT_LOCAL_MODELS[f'{self.model_name}_det']
        kpt_model_local = CTRL_ANNOT_LOCAL_MODELS[f'{self.model_name}_kpt']

        self.det_model = init_detector  (det_config_mmcv, det_model_local, self.device)
        self.kpt_model = init_pose_model(kpt_config_mmcv, kpt_model_local, self.device)

    def offload_model(self):
        raise NotImplementedError()

    def __call__(self, input_image: np.ndarray | Image.Image):

        if self.det_model is None \
        or self.kpt_model is None:
            self.load_model()

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        assert input_image.ndim == 3

        with torch.no_grad():
            image = torch.from_numpy(input_image).float().to(self.device)
            image = image / 255.0
            mmdet_results = inference_detector(human_det, image)
        
        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, self.person_id)
        
        with torch.no_grad():
            pose_results, _ = inference_top_down_pose_model(
                self.kpt_model,
                image,
                person_results,
                bbox_thr=self.bbox_thresh,
                format='xyxy',
                dataset=self.kpt_model.cfg.data['test']['type'],
                dataset_info=None,
                return_heatmap=False,
                outputs=None,   # use ('backbone', ) to return backbone feature
            )
            
        im_keypose_out = mmpose_visualize(
            image,
            pose_results,
            skeleton=self.skeleton,
            pose_kpt_color=self.pose_kpt_color,
            pose_link_color=self.pose_link_color,
            radius=2,
            thickness=2
        )
        im_keypose_out = im_keypose_out.astype(np.uint8)
        return im_keypose_out


#############################
#        Ez-DWPose          #
#############################
from .models.ezpose import DWposeDetector

class DWPoseEstimate:

    model_name = "dwpose"

    def __init__(self, device = DEVICE, preload: bool = False):
        self.device = str(device)
        self.model = None
        if preload:
            self.load_model()

    def load_model(self):
        for mtype in ['det', 'kpt']:
            model_name = f'{self.model_name}_{mtype}'
            local_model_path = CTRL_ANNOT_LOCAL_MODELS[model_name]
            if not os.path.exists(local_model_path):
                remote_model_path = CTRL_ANNOT_REMOTE_MODELS[model_name]
                load_file_from_url(remote_url=remote_model_path, 
                                    local_file=local_model_path)

        det_model_local = CTRL_ANNOT_LOCAL_MODELS[f'{self.model_name}_det']
        kpt_model_local = CTRL_ANNOT_LOCAL_MODELS[f'{self.model_name}_kpt']

        self.model = DWposeDetector(kpt_model_local, det_model_local, self.device)

    def offload_model(self):
        raise NotImplementedError()

    def __call__(self, input_image: np.ndarray | Image.Image):

        if self.model is None:
            self.load_model()

        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        kpt_out = self.model(input_image, output_type="pil", include_hands=True, include_face=True)
        return kpt_out


#############################
#           OpenPose        #
#############################
from .models.openpose import (
    detect_human_pose, draw_human_poses, 
    detect_animal_pose, draw_animal_poses,
)

class OpenposeEstimate:

    model_name = "openpose"

    def __init__(self, device = DEVICE, preload: bool = False, 
                                   include_body: bool = True, 
                                   include_hand: bool = True, 
                                   include_face: bool = True):
        self.device = device
        self.include_body = include_body    # Body model is always used in inference. Just ignore it in drawing (if False) 
        self.include_hand = include_hand
        self.include_face = include_face

        self.body_model = None
        self.hand_model = None
        self.face_model = None
        if preload:
            self.load_model()

    def load_model(self):
        all_models = ["body"]
        if self.include_hand:
            all_models.append("hand")
        if self.include_face:
            all_models.append("face")
        for mtype in all_models:
            model_name = f'{self.model_name}_{mtype}'
            local_model_path = CTRL_ANNOT_LOCAL_MODELS[model_name]
            if not os.path.exists(local_model_path):
                remote_model_path = CTRL_ANNOT_REMOTE_MODELS[model_name]
                load_file_from_url(remote_url=remote_model_path, 
                                    local_file=local_model_path)

        from .models.openpose import Body, Hand, Face
        self.body_model = Body(CTRL_ANNOT_LOCAL_MODELS[f"{self.model_name}_body"])
        self.hand_model = Hand(CTRL_ANNOT_LOCAL_MODELS[f"{self.model_name}_hand"]) if self.include_hand else None
        self.face_model = Face(CTRL_ANNOT_LOCAL_MODELS[f"{self.model_name}_face"]) if self.include_face else None

    def offload_model(self):
        if self.body_model is not None:
            self.body_model.model.to("cpu")
        if self.hand_model is not None:
            self.hand_model.model.to("cpu")
        if self.face_model is not None:
            self.face_model.model.to("cpu")

    def __call__(self, input_image: np.ndarray | Image.Image):

        if  self.body_model is None \
        and self.hand_model is None \
        and self.face_model is None :
            self.load_model()

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        H, W, _ = input_image.shape

        poses = detect_human_pose(input_image, self.body_model, self.face_model, self.hand_model, self.device)
        poses = draw_human_poses(poses, H, W, self.include_body, self.include_face, self.include_hand)
        return poses


class AnimalPoseEstimate:

    model_kpt = "animalpose"
    model_det = "dwpose_det"

    def __init__(self, device = DEVICE, preload: bool = False):
        self.device = device
        self.model = None
        if preload:
            self.load_model()

    def load_model(self):
        for model_name in [self.model_kpt, self.model_det]:
            local_model_path = CTRL_ANNOT_LOCAL_MODELS[model_name]
            if not os.path.exists(local_model_path):
                remote_model_path = CTRL_ANNOT_REMOTE_MODELS[model_name]
                load_file_from_url(remote_url=remote_model_path, 
                                    local_file=local_model_path)

        det_model_local = CTRL_ANNOT_LOCAL_MODELS[self.model_det]
        kpt_model_local = CTRL_ANNOT_LOCAL_MODELS[self.model_kpt]

        from .models.openpose import AnimalPose
        self.model = AnimalPose(det_model_local, kpt_model_local)

    def __call__(self, input_image):

        if  self.model is None:
            self.load_model()

        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        H, W, _ = input_image.shape

        poses = detect_animal_pose(input_image, self.model)
        poses = draw_animal_poses(poses, H, W)
        return poses


#############################
#       Universal           #
#############################
all_options_pose = ["densepose", "dwpose", "openpose", "animal_pose"]
if IS_MMPOSE_INSTALLED:
    all_options_pose.append("mmpose")

def apply_pose(input_image, model: str = "openpose", **kwargs):

    if model == "densepose":
        estimator = DensePoseEstimate()
        return estimator(input_image, **kwargs)

    elif model == "dwpose":
        estimator = DWPoseEstimate()
        return estimator(input_image)

    elif model == "openpose":
        estimator = OpenposeEstimate(**kwargs)
        return estimator(input_image)

    elif model == "animal_pose":
        estimator = AnimalPoseEstimate()
        return estimator(input_image)

    elif model == "mmpose": 
        if not IS_MMPOSE_INSTALLED:
            print("MMPose is not install")
            return input_image

        estimator = MMPoseEstimate()
        return estimator(input_image)

    else:
        raise ValueError(f"model = {model} is not supported!")
    

