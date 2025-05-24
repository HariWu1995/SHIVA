import os
import json

from packaging import version
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
from einops import rearrange

from .utils import save_video_frames
from .pipelines import load_pipeline
from .data.camera import Camera
from .data.trajectory import preprocess_traj


current_dir = Path(__file__).resolve().parent


def run_inference(
    image,
    prompt,
    device,
    cam_params,
    pipeline,

    num_steps: int = 20,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 3.0,

    image_width: int = 576, 
    image_height: int = 320,
    traj_height: int = 720,
    traj_width: int = 1280,  

    seed: int = 1995,
):
    # Preprocess
    plucker_embedding = preprocess_traj(image_width, image_height,
                                        traj_width, traj_height, cam_params, device)

    # Generate
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    with torch.no_grad():
        output = pipeline(
            image=image,
            pose_embedding=plucker_embedding,
            height=image_height,
            width=image_width,
            # num_frames=num_frames,
            num_inference_steps=num_steps,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            do_image_process=True,
            output_type='pt',
            generator=generator,
        ).frames[0].cpu()

    # Postprocess: 
    #   [L, 3, H, W] -> [L, H, W, 3]
    #         [0..1] -> [0..255]
    output = output.transpose(1, 2).transpose(2, 3)
    output = (output * 255).clamp(0, 255).to(torch.uint8)
    output = output.numpy()
    return output


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pipeline
    model_id = "svdxt"
    pipeline = load_pipeline(model_id, device)

    # Load trajectory
    trajectory_id = '0f47577ab3441480'
    trajectory_path = str(current_dir / f"pose_files/{trajectory_id}_{model_id}.txt").replace('\\', '/')
    with open(trajectory_path, 'r') as f:
        poses = f.readlines()
    poses = [pose.strip().split(' ') for pose in poses[1:]]
    cam_params = [Camera([float(x) for x in pose]) for pose in poses]

    # Inference
    samples = [
        "A_car_running_on_Mars.",
        "An_exploding_cheese_house.",
        "Leaves_are_falling_from_trees.",
        "Rocky_coastline_with_crashing_waves.",
        "Dolphins_leaping_out_of_the_ocean_at_sunset.",
        "A_tiny_finch_on_a_branch_with_spring_flowers_on_background.",
    ]

    for sample in samples:
        # Load template
        prompt = sample.replace('_', ' ')
        image = Image.open(f"./temp/target_images/{sample}.png")

        # Run pipeline
        output = run_inference(image, prompt, device, cam_params, pipeline)
        
        # Save
        output_path = f"./temp/{sample}{trajectory_id}_{model_id}.mp4"
        try:
            save_video_frames(output, output_path, revert_rgb=True, fps=5)
        except Exception:
            output_path = output_path.replace('.mp4', '.npy')
            with open(output_path, 'wb') as f:
                np.save(f, output)

