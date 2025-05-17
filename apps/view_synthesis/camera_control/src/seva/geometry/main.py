from typing import Literal
import numpy as np

import torch
import torch.nn.functional as F

from .preset import DEFAULT_FOV_RAD, get_default_intrinsics
from .trans import img2cam, cam2world
from .homog import to_hom
from .utils import viewmatrix


def get_camera_dist(
    source_c2ws: torch.Tensor,  # N x 3 x 4
    target_c2ws: torch.Tensor,  # M x 3 x 4
    mode: str = "translation",
):
    if mode == "rotation":
        dists = torch.acos((
            (
                torch.matmul(
                    source_c2ws[:, None, :3, :3],
                    target_c2ws[None, :, :3, :3].transpose(-1, -2)
                ).diagonal(offset=0, dim1=-2, dim2=-1).sum(-1) - 1
            ) / 2
        ).clamp(-1, 1)) * (180 / torch.pi)

    elif mode == "translation":
        dists = torch.norm(source_c2ws[:, None, :3, 3] \
                         - target_c2ws[None, :, :3, 3], dim=-1)

    else:
        raise NotImplementedError(f"Mode {mode} is not implemented for function `get_camera_dist`.")
    return dists


def get_image_grid(img_h, img_w):
    # add 0.5 is VERY important especially when your img_h and img_w
    # is not very large (e.g., 72)
    y_range = torch.arange(img_h, dtype=torch.float32).add_(0.5)
    x_range = torch.arange(img_w, dtype=torch.float32).add_(0.5)
    Y, X = torch.meshgrid(y_range, x_range, indexing="ij")          # [H,  W]
    xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)               # [HW, 2]
    xy_hom = to_hom(xy_grid)                                        # [HW, 3]
    return xy_hom


def get_center_and_ray(img_h, img_w, pose, intr):  # [HW,2]
    # given the intrinsic / extrinsic matrices, get the camera center and ray directions
    # assert(opt.camera.model=="perspective")

    # compute center and ray
    grid_img = get_image_grid(img_h, img_w)                         # [   HW, 3]
    grid_3D_cam = img2cam(grid_img.to(intr.device), intr.float())   # [B, HW, 3]
    center_3D_cam = torch.zeros_like(grid_3D_cam)                   # [B, HW, 3]

    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D_cam, pose)          # [B, HW, 3]
    center_3D = cam2world(center_3D_cam, pose)      # [B, HW, 3]
    ray = grid_3D - center_3D                       # [B, HW, 3]

    return center_3D, ray, grid_3D_cam


def get_plucker_coordinates(
    extrinsics_src,
    extrinsics,
    intrinsics=None,
    fov_rad=DEFAULT_FOV_RAD,
    target_size=[72, 72],
):
    if intrinsics is None:
        intrinsics = get_default_intrinsics(fov_rad).to(extrinsics.device)
    else:
        if not (torch.all(intrinsics[:, :2, -1] >= 0)
            and torch.all(intrinsics[:, :2, -1] <= 1)):
            intrinsics[:, :2] /= intrinsics.new_tensor(target_size).view(1, -1, 1) * 8
        # you should ensure the intrisics are expressed in
        # resolution-independent normalized image coordinates just performing a
        # very simple verification here checking if principal points are between 0 and 1
        assert (torch.all(intrinsics[:, :2, -1] >= 0)
            and torch.all(intrinsics[:, :2, -1] <= 1)), \
            "Intrinsics should be expressed in resolution-independent normalized image coordinates."

    c2w_src = torch.linalg.inv(extrinsics_src)

    # transform coordinates from source camera's coordinate system to coordinate system of respective camera
    extrinsics_rel = torch.einsum("vnm,vmp->vnp", extrinsics, c2w_src[None]\
                                          .repeat(extrinsics.shape[0], 1, 1))

    intrinsics[:, :2] *= extrinsics.new_tensor([target_size[1], 
                                                target_size[0]]).view(1, -1, 1)

    centers, rays, grid_cam = get_center_and_ray(
        img_h=target_size[0],
        img_w=target_size[1],
        pose=extrinsics_rel[:, :3, :],
        intr=intrinsics,
    )

    rays = F.normalize(rays, dim=-1)
    plucker = torch.cat((rays, torch.cross(centers, rays, dim=-1)), dim=-1)
    plucker = plucker.permute(0, 2, 1).reshape(plucker.shape[0], -1, *target_size)
    return plucker


def get_lookat(origins: torch.Tensor, viewdirs: torch.Tensor) -> torch.Tensor:
    """
    Triangulate a set of rays to find a single lookat point.

    Args:
        origins (torch.Tensor): A (N, 3) array of ray origins.
        viewdirs (torch.Tensor): A (N, 3) array of ray view directions.

    Returns:
        torch.Tensor: A (3,) lookat point.
    """
    viewdirs = F.normalize(viewdirs, dim=-1)
    eye = torch.eye(3, device=origins.device, dtype=origins.dtype)[None]

    # Calculate projection matrix I - rr^T
    I_min_cov = eye - (viewdirs[..., None] * viewdirs[..., None, :])
    
    # Compute sum of projections
    sum_proj = I_min_cov.matmul(origins[..., None]).sum(dim=-3)
    
    # Solve for the intersection point using least squares
    lookat = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]

    # Check NaNs.
    assert not torch.any(torch.isnan(lookat))
    return lookat


def poses_avg(poses):
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis   = poses[:, :3, 2].mean(0)
    up_vect  = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up_vect, position)
    return cam2world

