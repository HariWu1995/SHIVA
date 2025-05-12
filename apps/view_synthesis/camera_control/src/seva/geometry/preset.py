from typing import Literal

import torch
import numpy as np

from .w2cs import get_arc_horizontal_w2cs, get_lemniscate_w2cs, get_roll_w2cs, get_moving_w2cs
from .traj import generate_spiral_path


# Default: 54° = 0.9424777960769379
DEFAULT_FOV_RAD = np.deg2rad(54)


CAMERA_MOVEMENTS_PRESET = [
    "orbit",
    "spiral",
    "lemniscate",
    "zoom-in",
    "zoom-out",
    "dolly zoom-in",
    "dolly zoom-out",
    "move-forward",
    "move-backward",
    "move-up",
    "move-down",
    "move-left",
    "move-right",
    "roll",
]


def get_preset_pose_fov(
    option: Literal[CAMERA_MOVEMENTS_PRESET],
    num_frames: int,
    start_w2c: torch.Tensor,
    look_at: torch.Tensor,
    up_direction: torch.Tensor | None = None,
    fov: float = DEFAULT_FOV_RAD,
    spiral_radii: list[float] = [0.5, 0.5, 0.2],
    zoom_factor: float | None = None,
):
    poses = fovs = None

    if option == "orbit":
        poses = torch.linalg.inv(
            get_arc_horizontal_w2cs(
                start_w2c,
                look_at,
                up_direction,
                num_frames=num_frames,
                endpoint=False,
            )
        ).numpy()
        fovs = np.full((num_frames,), fov)
    
    elif option == "spiral":
        poses = generate_spiral_path(
            torch.linalg.inv(start_w2c)[None].numpy() @ np.diagflat([1, -1, -1, 1]),
            np.array([1, 5]),
            n_frames=num_frames,
            n_rots=2,
            zrate=0.5,
            radii=spiral_radii,
            endpoint=False,
        ) @ np.diagflat([1, -1, -1, 1])
        poses = np.concatenate([
            poses,
            np.array([0.0, 0.0, 0.0, 1.0])[None, None].repeat(len(poses), 0),
        ], axis=1)

        # We want the spiral trajectory to always start from start_w2c.
        # Thus, we apply the relative pose to get the final trajectory.
        poses = (np.linalg.inv(start_w2c.numpy())[None] @ np.linalg.inv(poses[:1]) @ poses)
        fovs = np.full((num_frames,), fov)
    
    elif option == "lemniscate":
        poses = torch.linalg.inv(
            get_lemniscate_w2cs(
                start_w2c,
                look_at,
                up_direction,
                num_frames,
                degree=60.0,
                endpoint=False,
            )
        ).numpy()
        fovs = np.full((num_frames,), fov)
    
    elif option == "roll":
        poses = torch.linalg.inv(
            get_roll_w2cs(
                start_w2c,
                look_at,
                None,
                num_frames,
                degree=360.0,
                endpoint=False,
            )
        ).numpy()
        fovs = np.full((num_frames,), fov)
    
    elif option in ["dolly zoom-in","dolly zoom-out","zoom-in","zoom-out",]:
        if option.startswith("dolly"):
            poses = torch.linalg.inv(
                get_moving_w2cs(
                    start_w2c,
                    look_at,
                    up_direction,
                    num_frames,
                    endpoint=True,
                    direction="backward" if option.endswith("zoom-in") else "forward",
                )
            ).numpy()
        else:
            poses = torch.linalg.inv(start_w2c)[None].repeat(num_frames, 1, 1).numpy()

        if zoom_factor is None:
            zoom_factor = 0.28 if option.endswith("zoom-in") else 1.5
        fov_rad_start = fov
        fov_rad_end = fov * zoom_factor
        fovs = np.linspace(0, 1, num_frames) * (fov_rad_end - fov_rad_start) + fov_rad_start
    
    elif option in ["move-forward","move-backward","move-up","move-down","move-left","move-right"]:
        poses = torch.linalg.inv(
            get_moving_w2cs(
                start_w2c,
                look_at,
                up_direction,
                num_frames,
                endpoint=True,
                direction=option.removeprefix("move-"),
            )
        ).numpy()
        fovs = np.full((num_frames,), fov)
    
    else:
        raise ValueError(f"Unknown preset option {option}.")

    return poses, fovs


def get_default_intrinsics(fov_rad = DEFAULT_FOV_RAD, aspect_ratio = 1.0):
    if not isinstance(fov_rad, torch.Tensor):
        fov_rad = torch.tensor([fov_rad] if isinstance(fov_rad, (int, float)) else fov_rad)
    
    # W >= H
    if aspect_ratio >= 1.0:
        focal_x = 0.5 / torch.tan(0.5 * fov_rad)
        focal_y = focal_x * aspect_ratio

    # W < H
    else:
        focal_y = 0.5 / torch.tan(0.5 * fov_rad)
        focal_x = focal_y / aspect_ratio

    device = focal_x.device
    dtype = focal_x.dtype

    I = torch.eye(3, device=device, dtype=bool)
    z = torch.ones_like(focal_x)

    intrinsics = focal_x.new_zeros((focal_x.shape[0], 3, 3))
    intrinsics[:, I] = torch.stack([focal_x, focal_y, z], dim=-1)
    intrinsics[:, :, -1] = torch.tensor([0.5, 0.5, 1.0], device=device, dtype=dtype)
    return intrinsics

