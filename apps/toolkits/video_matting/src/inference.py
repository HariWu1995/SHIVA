import os
from tqdm.auto import tqdm
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .model import MattingNetwork
from .utils import rvm_models, auto_downsample_ratio
from .io import VideoReader, VideoWriter
from .io import ImgSeqReader, ImgSeqWriter


def load_model(variant, device):
    assert variant in list(rvm_models.keys()), f"{variant} is not supported!"
    model = MattingNetwork(variant)
    model.load_state_dict(
            torch.load(rvm_models[variant], map_location=device))
    # model = torch.jit.script(model)
    # model = torch.jit.freeze(model)
    model.to(device=device)
    model.eval()
    return model


@torch.inference_mode()
def inference(
        model_variant: str,
        input_source: str,
        input_resize: Optional[Tuple[int, int]] = None,
        downsample_ratio: Optional[float] = None,
        output_alpha_mask : Optional[str] = None,
        output_composition: Optional[str] = None,
        output_foreground : Optional[str] = None,
        output_video_mbps: Optional[float] = None,
        output_type: str = 'video',
        num_workers: int = 0,
        seq_chunk: int = 1,
        progress: bool = True,
        device: Optional[torch.device] = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dtype: Optional[torch.dtype] = None,
    ):
    """
    Args:
        input_source: A video file, or an image sequence directory. 
                        Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. 
                            If not provided, model automatically set one.
        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. 
                File path if output_type == 'video'. 
                Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha_mask: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use > 0 for image input.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """
    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha_mask, output_foreground]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'
    
    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    elif os.path.isdir(input_source):
        source = ImgSeqReader(input_source, transform)
    else:
        raise OSError(f"{input_source} doesn't exist!")
    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)
    
    # Initialize writers
    if output_type == 'video':
        writer_kwargs = dict(
            frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30,
            frame_size = (source.frame_width, source.frame_height),
        )
        if output_composition is not None:
            writer_com = VideoWriter(path=output_composition, **writer_kwargs)
        if output_foreground is not None:
            writer_fgr = VideoWriter(path=output_foreground, **writer_kwargs)
        if output_alpha_mask is not None:
            writer_pha = VideoWriter(path=output_alpha_mask, **writer_kwargs)
    else:
        if output_composition is not None:
            writer_com = ImgSeqWriter(output_composition, 'png')
        if output_alpha_mask is not None:
            writer_pha = ImgSeqWriter(output_alpha_mask, 'png')
        if output_foreground is not None:
            writer_fgr = ImgSeqWriter(output_foreground, 'png')

    # Inference
    model = load_model(model_variant, device)
    model = model.eval()
    
    if (output_composition is not None) and (output_type == 'video'):
        bgr = torch.tensor([120, 255, 155]).div(255).view(1, 1, 3, 1, 1)
        bgr = bgr.to(device=device)
    
    try:
        bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
        rec = [None] * 4

        for src in reader:
            if src is None:
                continue
            if downsample_ratio is None:
                downsample_ratio = auto_downsample_ratio(*src.shape[2:])

            src = src.to(device=device, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
            fgr, pha, *rec = model(src, *rec, downsample_ratio)

            if output_foreground is not None:
                writer_fgr.write(fgr[0])
            if output_alpha_mask is not None:
                writer_pha.write(pha[0])
            if output_composition is not None:
                if output_type == 'video':
                    com = fgr * pha + bgr * (1 - pha)
                else:
                    fgr = fgr * pha.gt(0)
                    com = torch.cat([fgr, pha], dim=-3)
                writer_com.write(com[0])
            
            bar.update(src.size(1))
    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha_mask is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()


if __name__ == "__main__":

    input_video = "C:/Users/Mr. RIAH/Documents/Projects/SHIVA/_samples/footage/1917.mp4"
    output_video = "C:/Users/Mr. RIAH/Documents/Projects/SHIVA/temp/1917_{suffix}.mp4"

    inference(
        model_variant = "mobilenetv3",
        input_source = input_video,
        output_composition = output_video.format(suffix="composed"),
        output_foreground = output_video.format(suffix="fg"),
        output_alpha_mask = output_video.format(suffix="alpha"),
        output_type = "video",
    )
