import numpy as np

import torch
import torch.nn.functional as F


def get_lookat_w2cs(
     positions: torch.Tensor,
        lookat: torch.Tensor,
    up_vectors: torch.Tensor,
      face_off: bool = False,
):
    """
    Args:
        positions: (N, 3) tensor of camera positions
            lookat: (3,) tensor of lookat point
        up_vectors: (3,) or (N, 3) tensor of up vector

    Returns:
        w2cs: (N, 3, 3) tensor of world to camera rotation matrices
    """
    forward_vectors = F.normalize(lookat - positions, dim=-1)
    if face_off:
        forward_vectors = -forward_vectors
    if up_vectors.dim() == 1:
        up_vectors = up_vectors[None]
    right_vectors = F.normalize(torch.cross(forward_vectors, up_vectors, dim=-1), dim=-1)
    down_vectors = F.normalize(torch.cross(forward_vectors, right_vectors, dim=-1), dim=-1)
    Rs = torch.stack([right_vectors, down_vectors, forward_vectors], dim=-1)
    w2cs = torch.linalg.inv(rt_to_mat4(Rs, positions))
    return w2cs


def get_arc_horizontal_w2cs(
    ref_w2c: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor | None,
    num_frames: int,
    clockwise: bool = True,
    face_off: bool = False,
    endpoint: bool = False,
    degree: float = 360.0,
    ref_up_shift: float = 0.0,
    ref_radius_scale: float = 1.0,
    **_,
) -> torch.Tensor:

    device = ref_w2c.device
    ref_c2w = torch.linalg.inv(ref_w2c)
    ref_position = ref_c2w[:3, 3]
    
    if up is None:
        up = -ref_c2w[:3, 1]
    assert up is not None

    ref_position += up * ref_up_shift
    ref_position *= ref_radius_scale
    
    thetas = (
            torch.linspace(0.0, torch.pi * degree / 180, num_frames, device=device) if endpoint
        else torch.linspace(0.0, torch.pi * degree / 180, num_frames+1, device=device)[:-1]
    )
    if not clockwise:
        thetas = -thetas
    
    positions = (
        torch.einsum("nij,j->ni", 
                    roma.rotvec_to_rotmat(thetas[:, None] * up[None]),
                    ref_position - lookat)
        + lookat
    )
    return get_lookat_w2cs(positions, lookat, up, face_off=face_off)


def get_lemniscate_w2cs(
    ref_w2c: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor | None,
    num_frames: int,
    degree: float,
    endpoint: bool = False,
    **_,
) -> torch.Tensor:
    
    device = ref_w2c.device
    ref_c2w = torch.linalg.inv(ref_w2c)
    angle = torch.linalg.norm(ref_c2w[:3, 3] - lookat) * np.tan(degree / 360 * np.pi)

    # Lemniscate curve in camera space. Starting at the origin.
    thetas = (
             torch.linspace(0, 2 * torch.pi, num_frames, device=device) if endpoint
        else torch.linspace(0, 2 * torch.pi, num_frames+1, device=device)[:-1]
    ) + torch.pi / 2

    positions = torch.stack([
            angle * torch.cos(thetas)                     / (1 + torch.sin(thetas) ** 2),
            angle * torch.cos(thetas) * torch.sin(thetas) / (1 + torch.sin(thetas) ** 2),
                    torch.zeros(num_frames, device=device),
        ], dim=-1)

    # Transform to world space.
    positions = torch.einsum("ij,nj->ni", ref_c2w[:3], F.pad(positions, (0, 1), value=1.0))

    if up is None:
        up = -ref_c2w[:3, 1]
    assert up is not None
    return get_lookat_w2cs(positions, lookat, up)


def get_moving_w2cs(
    ref_w2c: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor | None,
    num_frames: int,
    endpoint: bool = False,
    direction: str = "forward",
    tilt_xy: torch.Tensor = None,
):
    """
    Args:
        ref_w2c: (4, 4) tensor of the reference wolrd-to-camera matrix
        lookat: (3,) tensor of lookat point
        up: (3,) tensor of up vector

    Returns:
        w2cs: (N, 3, 3) tensor of world to camera rotation matrices
    """
    ref_c2w = torch.linalg.inv(ref_w2c)
    ref_position = ref_c2w[:3, -1]
    if up is None:
        up = -ref_c2w[:3, 1]

    direction_vectors = {
         "forward":  (lookat - ref_position).clone(),
        "backward": -(lookat - ref_position).clone(),
            "up":  up.clone(),
          "down": -up.clone(),
        "right":  torch.cross((lookat - ref_position), up, dim=0),
         "left": -torch.cross((lookat - ref_position), up, dim=0),
    }
    if direction not in direction_vectors:
        raise ValueError(
            f"Invalid direction: {direction}. Must be one of {list(direction_vectors.keys())}"
        )

    device = ref_w2c.device
    positions = ref_position + (
        F.normalize(direction_vectors[direction], dim=0)
        * (
                torch.linspace(0, .99, num_frames, device=device) if endpoint
            else torch.linspace(0, 1, num_frames+1, device=device)[:-1]
        )[:, None]
    )

    if tilt_xy is not None:
        positions[:, :2] += tilt_xy

    return get_lookat_w2cs(positions, lookat, up)


def get_roll_w2cs(
    ref_w2c: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor | None,
    num_frames: int,
    endpoint: bool = False,
    degree: float = 360.0,
    **_,
) -> torch.Tensor:

    ref_c2w = torch.linalg.inv(ref_w2c)
    ref_position = ref_c2w[:3, 3]
    if up is None:
        up = -ref_c2w[:3, 1]  # Infer the up vector from the reference.

    # Create vertical angles
    device = ref_w2c.device
    thetas = (
             torch.linspace(0.0, torch.pi * degree / 180, num_frames, device=device) if endpoint
        else torch.linspace(0.0, torch.pi * degree / 180, num_frames+1, device=device)[:-1]
    )[:, None]

    lookat_vector = F.normalize(lookat[None].float(), dim=-1)
    lookatup_vector = torch.einsum("ij,ij->i", lookat_vector, up)[:, None]

    up = up[None]
    up = (
                                     up   *      torch.cos(thetas)
        + torch.cross(lookat_vector, up)  *      torch.sin(thetas)
        + lookat_vector * lookatup_vector * (1 - torch.cos(thetas))
    )

    # Normalize the camera orientation
    return get_lookat_w2cs(ref_position[None].repeat(num_frames, 1), lookat, up)


