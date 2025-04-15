import os
import cv2
import numpy as np
import torch


def preprocess(image, device):
    # Resize
    scale = 640 / max(image.shape[:2])
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array([104.008, 116.669, 122.675])

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)

    return image, raw_image


def visualize(
    image,
    pose_result,
    skeleton=None,
    kpt_score_thresh=0.1,
    pose_kpt_color=None,
    pose_link_color=None,
    radius=4,
    thickness=1
):
    """
    Draw keypoints and links on an image.

    Args:
        image (ndarry): The image to draw poses on.
        pose_result (list[kpts]): The poses to draw. Each element kpts is
                                    a set of K keypoints as an Kx3 numpy.ndarray, where each
                                    keypoint is represented as x, y, score.
        kpt_score_thresh (float, optional): Minimum score of keypoints to be shown. 
                                            Default: 0.3.
        pose_kpt_color (np.array[Nx3]`): Color of N keypoints. 
                                            If None, the keypoint will not be drawn.
        pose_link_color (np.array[Mx3]): Color of M links. 
                                            If None, the links will not be drawn.
        thickness (int): Thickness of lines.
    """
    H, W, _ = image.shape
    image = np.zeros(image.shape)

    for idx, kpts in enumerate(pose_result):
        if idx > 1:
            continue
        kpts = kpts['keypoints']
        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)

            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

                # skip the point that should not be drawn
                if kpt_score < kpt_score_thresh or pose_kpt_color[kid] is None:
                    continue

                color = tuple(int(c) for c in pose_kpt_color[kid])
                cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, -1)

        # draw links
        if pose_link_color is not None \
              and skeleton is not None:
            assert len(pose_link_color) == len(skeleton)

            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                # skip the link that should not be drawn
                if (pos1[0] <= 0 or pos1[0] >= W \
                 or pos1[1] <= 0 or pos1[1] >= H \
                 or pos2[0] <= 0 or pos2[0] >= W \
                 or pos2[1] <= 0 or pos2[1] >= H \
                 or kpts[sk[0], 2] < kpt_score_thresh \
                 or kpts[sk[1], 2] < kpt_score_thresh \
                 or pose_link_color[sk_id] is None):
                    continue

                color = tuple(int(c) for c in pose_link_color[sk_id])
                cv2.line(image, pos1, pos2, color, thickness=thickness)

    return image


