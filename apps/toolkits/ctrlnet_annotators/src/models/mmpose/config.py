from pathlib import Path

MMPOSE_DIR = Path(__file__).resolve().parents[0]

det_config_file = str(MMPOSE_DIR / 'faster_rcnn_r50_fpn_coco.py')
kpt_config_file = str(MMPOSE_DIR / 'hrnet_w48_coco_256x192.py')

det_cat_id = 1  # 0: background, 1: person
bbox_thresh = 0.2


# Reference: https://mmpose.readthedocs.io/en/latest/guide_to_framework.html

body_parts = {
    "0": "Nose",
    "1": "Left Eye",
    "2": "Right Eye",
    "3": "Left Ear",
    "4": "Right Ear",
    "5": "Left Shoulder",
    "6": "Right Shoulder",
    "7": "Left Elbow",
    "8": "Right Elbow",
    "9": "Left Wrist",
    "10": "Right Wrist",
    "11": "Left Hip",
    "12": "Right Hip",
    "13": "Left Knee",
    "14": "Right Knee",
    "15": "Left Ankle",
    "16": "Right Ankle"
}

skeleton = [
    [15, 13], [13, 11],         #  (Left) Ankle -> Knee -> Hip
    [16, 14], [14, 12],         # (Right) Ankle -> Knee -> Hip
    [11, 12], 
    [5, 11],                    #  (Left) Shoulder -> Hip
    [6, 12],                    # (Right) Shoulder -> Hip
    [5, 6], 
    [5, 7], 
    [6, 8],
    [7, 9], [8, 10],
    [1, 2], [0, 1], [0, 2], 
    [1, 3], [2, 4], [3, 5], [4, 6]
]

# Corresponding to skeleton (above)
pose_link_color = [
    [0, 255, 0], 
    [0, 255, 0], 
    [255, 128, 0], 
    [255, 128, 0],
    [51, 153, 255], 
    [51, 153, 255], 
    [51, 153, 255], 
    [51, 153, 255], 
    [0, 255, 0],
    [255, 128, 0],
    [0, 255, 0], 
    [255, 128, 0], 
    [51, 153, 255], 
    [51, 153, 255], 
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255], 
    [51, 153, 255], 
    [51, 153, 255],
]

pose_kpt_color = [
    [51, 153, 255], 
    [51, 153, 255], 
    [51, 153, 255], 
    [51, 153, 255], 
    [51, 153, 255],
    [0, 255, 0],
    [255, 128, 0], 
    [0, 255, 0],
    [255, 128, 0], 
    [0, 255, 0], 
    [255, 128, 0], 
    [0, 255, 0],
    [255, 128, 0],
    [0, 255, 0], 
    [255, 128, 0], 
    [0, 255, 0], 
    [255, 128, 0],
]

