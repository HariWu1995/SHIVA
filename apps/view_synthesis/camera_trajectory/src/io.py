import os
import imageio.v3 as iio

import numpy as np
import torch


def save_output(
    samples,
    save_path,
    video_save_fps: int = 2,
):
    if os.path.isdir(save_path) is False:
        os.makedirs(save_path)

    for sample in samples:
        media_type = "video"
        if "/" in sample:
            sample_, media_type = sample.split("/")
        else:
            sample_ = sample

        value = samples[sample]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        else:
            value = torch.tensor(value)

        if media_type == "image":
            value = (value.permute(0, 2, 3, 1) + 1) / 2.0
            value = (value * 255).clamp(0, 255).to(torch.uint8)
            iio.imwrite(
                os.path.join(save_path, f"{sample_}.mp4") if sample_ else f"{save_path}.mp4",
                value,
                fps=video_save_fps,
                macro_block_size=1,
                ffmpeg_log_level="error",
            )

            sample_dir = s.path.join(save_path, sample_)
            if os.path.isdir(sample_dir) is False:
                os.makedirs(sample_dir)
            for i, s in enumerate(value):
                iio.imwrite(os.path.join(sample_dir, f"{i:03d}.png"), s)

        elif media_type == "video":
            value = (value.permute(0, 2, 3, 1) + 1) / 2.0
            value = (value * 255).clamp(0, 255).to(torch.uint8)
            iio.imwrite(
                os.path.join(save_path, f"{sample_}.mp4"),
                value,
                fps=video_save_fps,
                macro_block_size=1,
                ffmpeg_log_level="error",
            )
       
        elif media_type == "raw":
            torch.save(value, os.path.join(save_path, f"{sample_}.pt"))

        else:
            print(f"`media_type` = {media_type} is not supported! This step is skipped!")


def decode_output(
    samples,
    T,
    indices=None,
):
    # decode model output into dict if it is not
    if isinstance(samples, dict):
        # model with postprocessor and outputs dict
        for sample, value in samples.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            elif isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            else:
                value = torch.tensor(value)
            if value.shape[0] == T and indices is not None:
                value = value[indices]
            samples[sample] = value
    else:
        # model without postprocessor and outputs tensor (rgb)
        samples = samples.detach().cpu()
        if indices is not None and samples.shape[0] == T:
            samples = samples[indices]
        samples = {"samples-rgb/image": samples}
    return samples

