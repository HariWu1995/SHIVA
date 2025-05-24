import torch
import numpy as np

from .camera import get_relative_pose, ray_condition


def preprocess_traj(
    image_width, image_height,
     traj_width,  traj_height,
     cam_params,  device,
):
    image_wh_ratio = image_width / image_height
    traj_wh_ratio  =  traj_width /  traj_height

    if traj_wh_ratio > image_wh_ratio:
        resized_width = image_height * traj_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_width * cam_param.fx / image_width
    else:
        resized_height = image_width / traj_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_height * cam_param.fy / image_height

    intrinsic = np.asarray([[
            cam_param.fx * image_width, cam_param.fy * image_height,
            cam_param.cx * image_width, cam_param.cy * image_height]
        for cam_param in cam_params
    ], dtype=np.float32)

    K = torch.as_tensor(intrinsic)[None]        # [1, 1, 4]
    c2ws = get_relative_pose(cam_params, True)
    c2ws = torch.as_tensor(c2ws)[None]          # [1, n_frame, 4, 4]

    plucker_embedding = ray_condition(K, c2ws, image_height, image_width, device='cpu')       # b f h w 6
    plucker_embedding = plucker_embedding.permute(0, 1, 4, 2, 3).contiguous().to(device=device)

    return plucker_embedding

