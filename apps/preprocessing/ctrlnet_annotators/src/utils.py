"""
Reference:
    https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite
"""
import os
from pathlib import Path

import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_ROOT = os.environ.get('SHIVA_CKPT_ROOT', None)
if CHECKPOINT_ROOT is not None:
    CTRL_ANNOT_DIR = Path(CHECKPOINT_ROOT) / 'ctrlnet_annotators'
else:
    CTRL_ANNOT_DIR = Path(__file__).parents[4] / 'checkpoints/ctrlnet_annotators'

if os.path.isdir(CTRL_ANNOT_DIR) is False:
    os.makedirs(CTRL_ANNOT_DIR)


def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


def load_file_from_url(
    remote_url: str,
    model_dir: str | None = None,
    local_file: str | None = None,
    hash_prefix: str | None = None,
    progress: bool = True,
) -> str:
    """
    Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    assert (model_dir is not None) \
        or (local_file is not None), "Either `model_dir` or `local_file` must be not None."
    if not local_file:
        if os.path.isdir(model_dir) is False:
            os.makedirs(model_dir)
        from urllib.parse import urlparse
        parts = urlparse(remote_url)
        local_file = os.path.basename(parts.path)
        local_file = os.path.abspath(os.path.join(model_dir, local_file))
    if os.path.isfile(local_file) is False:
        print(f'Downloading: "{remote_url}" to {local_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(remote_url, local_file, progress=progress, hash_prefix=hash_prefix)
    return local_file


CTRL_ANNOT_LOCAL_MODELS = {
               "hed": str(CTRL_ANNOT_DIR / "ControlNetHED.pth"),
        "anifaceseg": str(CTRL_ANNOT_DIR / "anime-face-segment.pth"),
        "animalpose": str(CTRL_ANNOT_DIR / "anime-pose-keypoints.pth"),
         "densepose": str(CTRL_ANNOT_DIR / "densepose_r50_fpn_dl.torchscript"),
        "mmpose_det": str(CTRL_ANNOT_DIR / "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"),
        "mmpose_kpt": str(CTRL_ANNOT_DIR / "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"),
        "dwpose_det": str(CTRL_ANNOT_DIR / "yolox_l.onnx"),
        "dwpose_kpt": str(CTRL_ANNOT_DIR / "dw-ll_ucoco_384.onnx"),
    # "openpose_det": str(CTRL_ANNOT_DIR / "yolox_l.onnx"),
    # "openpose_kpt": str(CTRL_ANNOT_DIR / "dw-ll_ucoco_384.onnx"),
     "openpose_face": str(CTRL_ANNOT_DIR / "face_openpose.pth"),
     "openpose_hand": str(CTRL_ANNOT_DIR / "hand_openpose.pth"),
     "openpose_body": str(CTRL_ANNOT_DIR / "body_openpose.pth"),
             "leres": str(CTRL_ANNOT_DIR / "leres101.pth"),
           "pix2pix": str(CTRL_ANNOT_DIR / "pix2pix_net_G.pth"),
           "lineart": str(CTRL_ANNOT_DIR / "line_art.pth"),
        "linecoarse": str(CTRL_ANNOT_DIR / "line_coarse.pth"),
         "lineanime": str(CTRL_ANNOT_DIR / "line_anime.pth"),
         "linemanga": str(CTRL_ANNOT_DIR / "line_manga.pth"),
         "dpt_large": str(CTRL_ANNOT_DIR / "dpt_large-midas.pth"),
        "dpt_hybrid": str(CTRL_ANNOT_DIR / "dpt_hybrid-midas.pth"),
         "midas_v21": str(CTRL_ANNOT_DIR / "midas_v21.pth"),
        "midas_v21s": str(CTRL_ANNOT_DIR / "midas_v21_small.pth"),
        "mlsd_large": str(CTRL_ANNOT_DIR / "mlsd_large_512_fp32.pth"),
         "mlsd_tiny": str(CTRL_ANNOT_DIR / "mlsd_tiny_512_fp32.pth"),
         "normalbae": str(CTRL_ANNOT_DIR / "normalbae_scannet.pth"),
             "dsine": str(CTRL_ANNOT_DIR / "dsine.pth"),
           "pidinet": str(CTRL_ANNOT_DIR / "table5_pidinet.pth"),
              "teed": str(CTRL_ANNOT_DIR / "TEED.pth"),
             "mteed": str(CTRL_ANNOT_DIR / "MTEED.pth"),
               "zoe": str(CTRL_ANNOT_DIR / "ZoeD_M12_N.pt"),
}

CTRL_ANNOT_REMOTE_MODELS = {
         "densepose": "https://huggingface.co/LayerNorm/DensePose-TorchScript-with-hint-image/resolve/main/densepose_r50_fpn_dl.torchscript",
        "mmpose_det": "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        "mmpose_kpt": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
        "dwpose_det": "https://huggingface.co/RedHash/DWPose/resolve/main/yolox_l.onnx",
        "dwpose_kpt": "https://huggingface.co/RedHash/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
    # "openpose_det": "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
    # "openpose_kpt": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
     "openpose_face": "https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth",
     "openpose_hand": "https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth",
     "openpose_body": "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth",
               "hed": "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth",
             "leres": "https://huggingface.co/lllyasviel/Annotators/resolve/main/res101.pth",
           "pix2pix": "https://huggingface.co/lllyasviel/Annotators/resolve/main/latest_net_G.pth",
           "lineart": "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth",
        "linecoarse": "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth",
         "lineanime": "https://huggingface.co/lllyasviel/Annotators/resolve/main/netG.pth",
         "linemanga": "https://huggingface.co/lllyasviel/Annotators/resolve/main/erika.pth",
        "mlsd_large": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth",
         "mlsd_tiny": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_tiny_512_fp32.pth",
         "normalbae": "https://huggingface.co/lllyasviel/Annotators/resolve/main/scannet.pt",
           "pidinet": "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth",
               "zoe": "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt",
             "dsine": "https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/dsine.pt",
        "animalpose": "https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.onnx",
        "anifaceseg": "https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/UNet.pth",
              "teed": "https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/7_model.pth",
             "mteed": "https://huggingface.co/TheMistoAI/MistoLine/resolve/main/Anyline/MTEED.pth",
        "dpt_hybrid": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/dpt_hybrid-midas-501f0c75.pt",
         "dpt_large": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/dpt_large-midas-2f21e586.pt",
         "midas_v21": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
        "midas_v21s": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
}

# if os.environ.get('SHIVA_CKPT_PRELOAD', False):
if True:
    for model_name, model_path in CTRL_ANNOT_LOCAL_MODELS.items():
        if os.path.isfile(model_path):
            continue
        load_file_from_url(remote_url=CTRL_ANNOT_REMOTE_MODELS[model_name], 
                           local_file=CTRL_ANNOT_LOCAL_MODELS[model_name])


