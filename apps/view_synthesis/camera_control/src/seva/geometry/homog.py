import torch


def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom


def to_hom_pose(pose):
    # get homogeneous coordinates of the input pose
    if pose.shape[-2:] == (3, 4):
        pose_hom = torch.eye(4, device=pose.device)[None].repeat(pose.shape[0], 1, 1)
        pose_hom[:, :3, :] = pose
        return pose_hom
    return pose

