import os
import json
import argparse

from tqdm import tqdm

import numpy as np
import torch

from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open

from ..path import CAMCTRL_LOCAL_MODELS
from .data import Camera
from .utils import save_videos_grid, setup_for_distributed
from .pipelines import load_pipeline


def main(args):
    os.makedirs(args.out_root, exist_ok=True)
    rank = args.local_rank
    setup_for_distributed(rank == 0)

    gpu_id = rank % torch.cuda.device_count()

    model_configs = OmegaConf.load(args.model_config)
    unet_additional_kwargs = model_configs['unet_additional_kwargs'] if 'unet_additional_kwargs' in model_configs else None
    noise_scheduler_kwargs = model_configs['noise_scheduler_kwargs']
    attn_processor_kwargs = model_configs['attention_processor_kwargs']
    pose_encoder_kwargs = model_configs['pose_encoder_kwargs']

    print(f'Constructing pipeline')
    device = torch.device(f"cuda:{gpu_id}")

    print('Loading K, R, t matrix')

    with open(args.trajectory_file, 'r') as f:
        poses = f.readlines()
    poses = [pose.strip().split(' ') for pose in poses[1:]]

    cam_params = [[float(x) for x in pose] for pose in poses]
    cam_params = [Camera(cam_param) for cam_param in cam_params]

    sample_wh_ratio = args.image_width / args.image_height
    pose_wh_ratio = args.original_pose_width / args.original_pose_height

    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = args.image_height * pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / args.image_width
    else:
        resized_ori_h = args.image_width / pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / args.image_height

    intrinsic = np.asarray([[cam_param.fx * args.image_width,
                             cam_param.fy * args.image_height,
                             cam_param.cx * args.image_width,
                             cam_param.cy * args.image_height]
                            for cam_param in cam_params], dtype=np.float32)

    K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
    c2ws = get_relative_pose(cam_params)
    c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]

    plucker_embedding = ray_condition(K, c2ws, args.image_height, args.image_width, device='cpu')[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
    plucker_embedding = plucker_embedding[None].to(device)  # B V 6 H W
    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")

    if args.visualization_captions.endswith('.json'):
        json_file = json.load(open(args.visualization_captions, 'r'))
        captions = json_file['captions'] if 'captions' in json_file else json_file['prompts']
        if args.use_negative_prompt:
            negative_prompts = json_file['negative_prompts']
        else:
            negative_prompts = None
        if isinstance(captions[0], dict):
            captions = [cap['caption'] for cap in captions]
        if args.use_specific_seeds:
            specific_seeds = json_file['seeds']
        else:
            specific_seeds = None
    elif args.visualization_captions.endswith('.txt'):
        with open(args.visualization_captions, 'r') as f:
            captions = f.readlines()
        captions = [cap.strip() for cap in captions]
        negative_prompts = None
        specific_seeds = None

    N = int(len(captions) // args.n_workers)
    remainder = int(len(captions) % args.n_workers)
    prompts_per_gpu = [N + 1 if gpu_id < remainder else N for gpu_id in range(args.n_workers)]

    low_idx = sum(prompts_per_gpu[:gpu_id])
    high_idx = low_idx + prompts_per_gpu[gpu_id]

    prompts = captions[low_idx: high_idx]
    negative_prompts = negative_prompts[low_idx: high_idx] if negative_prompts is not None else None

    specific_seeds = specific_seeds[low_idx: high_idx] if specific_seeds is not None else None

    print(f"rank {rank} / {torch.cuda.device_count()}, number of prompts: {len(prompts)}")

    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    for local_idx, caption in tqdm(enumerate(prompts)):
        if specific_seeds is not None:
            specific_seed = specific_seeds[local_idx]
            generator.manual_seed(specific_seed)

        sample = pipeline(
            prompt=caption,
            negative_prompt=negative_prompts[local_idx] if negative_prompts is not None else None,
            pose_embedding=plucker_embedding,
            video_length=args.video_length,
            height=args.image_height,
            width=args.image_width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).videos  # [1, 3, f, h, w]

        save_name = "_".join(caption.split(" "))
        save_videos_grid(sample, f"{args.out_root}/{save_name}.mp4")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str)

    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=384)
    parser.add_argument("--video_length", type=int, default=16)

    # Model config
    parser.add_argument("--finetuned_model", default=None)
    parser.add_argument("--base_model_path", type=str, help='path to the sd model folder')
    parser.add_argument("--unet_subfolder", type=str, help='subfolder name of unet ckpt')
    parser.add_argument("--model_config", type=str)
    
    parser.add_argument("--motion_module_ckpt", type=str, help='path to the animatediff motion module ckpt')
    parser.add_argument("--pose_adaptor_ckpt", default=None, help='path to the camera control model ckpt')
    parser.add_argument("--image_lora_ckpt", default=None)
    parser.add_argument("--image_lora_rank", default=2, type=int)
    
    parser.add_argument("--original_pose_width", type=int, default=1280, help='the width of the video used to extract camera trajectory')
    parser.add_argument("--original_pose_height", type=int, default=720, help='the height of the video used to extract camera trajectory')
    
    # Inference config
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=14.0)
    parser.add_argument("--trajectory_file", required=True, help='txt file')
    
    parser.add_argument("--visualization_captions", required=True, help='prompts path, json or txt')
    parser.add_argument("--use_negative_prompt", action='store_true', help='whether to use negative prompts')
    parser.add_argument("--use_specific_seeds", action='store_true', help='whether to use specific seeds for each prompt')

    # Parallelism & DDP
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--world_size", type=int, default=1, help="number of the distributed processes.")
    parser.add_argument('--local_rank', type=int, default=-1, help='Replica rank on the current node. This field is required by `torch.distributed.launch`.')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    # Load pipeline
    pipeline = load_pipeline(
        base_model_path = CAMCTRL_LOCAL_MODELS["sd15/camctrl"], 
        image_lora_ckpt = None, 
        image_lora_rank = 2, 
        pose_adaptor_ckpt, 
        unet_mm_ckpt, 
        unet_subfolder: str = 'unet', 
        unet_extra_kwargs = dict(),
        pose_encoder_kwargs = dict(), 
        noise_scheduler_kwargs = dict(), 
        attention_processor_kwargs = dict(),
        finetuned_model = None, 
        gpu_id = 0,
    )

    args = get_arguments()
    main(args)
