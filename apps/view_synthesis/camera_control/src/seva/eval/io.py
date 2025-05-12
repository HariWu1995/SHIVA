import os
import torch
import imageio.v3 as iio


def save_output(samples, save_path, video_save_fps: int = 2):
    os.makedirs(save_path, exist_ok=True)

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

            os.makedirs(os.path.join(save_path, sample_), exist_ok=True)
            for i, s in enumerate(value):
                iio.imwrite(os.path.join(save_path, sample_, f"{i:03d}.png"), s)

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
            pass


