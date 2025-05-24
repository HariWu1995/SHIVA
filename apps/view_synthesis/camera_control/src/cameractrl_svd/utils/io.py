import os
from typing import List

import imageio
import numpy as np
import cv2

import torch
import torchvision
from einops import rearrange


def save_video_frames(frames: List[np.ndarray], path: str, revert_rgb: bool = True, fps=8):
    height, width = frames[0].shape[:2]

    if path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in frames:
        assert isinstance(frame, np.ndarray)
        if revert_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_writer.write(frame)
    
    video_writer.release()


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


