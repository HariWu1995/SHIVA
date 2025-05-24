import os
import json

from decord import VideoReader

import random as rd
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from .camera import Camera, ray_condition
from .transform import RandomHorizontalFlipWithPose


class RealEstate10K(Dataset):

    def __init__(
        self,
        root_path,
        annotation_json,
        sample_stride=4,
        sample_n_frames=16,
        sample_size=[256, 384],
        is_image=False,
    ):
        self.root_path = root_path
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image

        self.dataset = json.load(open(os.path.join(root_path, annotation_json), 'r'))
        self.length = len(self.dataset)

        if isinstance(sample_size, int):
            sample_size = (sample_size, sample_size)
        else:
            sample_size = tuple(sample_size)

        pixel_transforms = [
            transforms.Resize(sample_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]

        self.pixel_transforms = transforms.Compose(pixel_transforms)

    def load_video_reader(self, idx):
        video_dict = self.dataset[idx]
        video_path = os.path.join(self.root_path, video_dict['clip_path'])
        video_reader = VideoReader(video_path)
        return video_reader, video_dict['caption']

    def get_batch(self, idx):
        video_reader, video_caption = self.load_video_reader(idx)
        total_frames = len(video_reader)

        if self.is_image:
            frame_indice = [rd.randint(0, total_frames - 1)]
        else:
            if isinstance(self.sample_stride, int):
                current_sample_stride = self.sample_stride
            else:
                assert len(self.sample_stride) == 2
                assert  (self.sample_stride[0] >= 1) \
                    and (self.sample_stride[1] >= self.sample_stride[0])
                current_sample_stride = rd.randint(self.sample_stride[0], self.sample_stride[1])

            cropped_length = self.sample_n_frames * current_sample_stride
            start_frame_ind = rd.randint(0, max(0, total_frames - cropped_length - 1))
            end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

            assert end_frame_ind - start_frame_ind >= self.sample_n_frames
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)

        pixel_values = torch.from_numpy(video_reader.get_batch(frame_indice).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.

        if self.is_image:
            pixel_values = pixel_values[0]

        return pixel_values, video_caption

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                video, video_caption = self.get_batch(idx)
                break

            except Exception as e:
                idx = rd.randint(0, self.length - 1)

        video = self.pixel_transforms(video)
        sample = dict(pixel_values=video, caption=video_caption)

        return sample


class RealEstate10KPose(Dataset):

    def __init__(
        self,
        root_path,
        annotation_json,
        sample_stride=4,
        minimum_sample_stride=1,
        sample_n_frames=16,
        relative_pose=False,
        zero_t_first_frame=False,
        sample_size=[256, 384],
        rescale_fxy=False,
        shuffle_frames=False,
        use_flip=False,
        return_clip_name=False,
    ):
        self.root_path = root_path
        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        self.sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = sample_n_frames
        self.return_clip_name = return_clip_name

        self.dataset = json.load(open(os.path.join(root_path, annotation_json), 'r'))
        self.length = len(self.dataset)

        if isinstance(sample_size, int) is False:
            sample_size = tuple(sample_size)  
        else:
            sample_size = (sample_size, sample_size)
        self.sample_size = sample_size

        if use_flip:
            pixel_transforms = [
                transforms.Resize(sample_size),
                RandomHorizontalFlipWithPose(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        else:
            pixel_transforms = [
                transforms.Resize(sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        
        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = sample_size[1] / sample_size[0]

        self.pixel_transforms = pixel_transforms
        self.shuffle_frames = shuffle_frames
        self.use_flip = use_flip

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]

        source_cam_c2w = abs_c2ws[0]
        if self.zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])

        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        abs2rel = target_cam_c2w @ abs_w2cs[0]

        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def load_video_reader(self, idx):
        video_dict = self.dataset[idx]
        video_path = os.path.join(self.root_path, video_dict['clip_path'])
        video_reader = VideoReader(video_path)
        return video_dict['clip_name'], video_reader, video_dict['caption']

    def load_cameras(self, idx):
        video_dict = self.dataset[idx]
        pose_file = os.path.join(self.root_path, video_dict['pose_file'])
        with open(pose_file, 'r') as f:
            poses = f.readlines()
        poses = [pose.strip().split(' ') for pose in poses[1:]]
        cam_params = [[float(x) for x in pose] for pose in poses]
        cam_params = [Camera(cam_param) for cam_param in cam_params]
        return cam_params

    def get_batch(self, idx):
        clip_name, video_reader, video_caption = self.load_video_reader(idx)

        cam_params = self.load_cameras(idx)
        total_frames = len(cam_params)

        assert total_frames >= self.sample_n_frames
        current_sample_stride = self.sample_stride

        if total_frames < self.sample_n_frames * current_sample_stride:
            maximum_sample_stride = int(total_frames // self.sample_n_frames)
            current_sample_stride = rd.randint(self.minimum_sample_stride, maximum_sample_stride)

        cropped_length = self.sample_n_frames * current_sample_stride
        start_frame_ind = rd.randint(0, max(0, total_frames - cropped_length - 1))
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        assert end_frame_ind - start_frame_ind >= self.sample_n_frames
        frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)

        condition_image_ind = rd.sample(list(set(range(total_frames)) - set(frame_indices.tolist())), 1)
        condition_image = torch.from_numpy(video_reader.get_batch(condition_image_ind).asnumpy()).permute(0, 3, 1, 2).contiguous()
        condition_image = condition_image / 255.

        if self.shuffle_frames:
            perm = np.random.permutation(self.sample_n_frames)
            frame_indices = frame_indices[perm]

        pixel_values = torch.from_numpy(video_reader.get_batch(frame_indices).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.

        cam_params = [cam_params[indice] for indice in frame_indices]
        if self.rescale_fxy:
            ori_h, ori_w = pixel_values.shape[-2:]
            ori_wh_ratio = ori_w / ori_h
            
            # rescale fx
            if ori_wh_ratio > self.sample_wh_ratio:
                resized_ori_w = self.sample_size[0] * ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
            
            # rescale fy
            else:
                resized_ori_h = self.sample_size[1] / ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]

        intrinsics = np.asarray([[
                cam_param.fx * self.sample_size[1],
                cam_param.fy * self.sample_size[0],
                cam_param.cx * self.sample_size[1],
                cam_param.cy * self.sample_size[0]]
            for cam_param in cam_params
        ], dtype=np.float32)

        intrinsics = torch.as_tensor(intrinsics)[None]      # [1, n_frame, 4]

        if self.relative_pose:
            c2w_poses = self.get_relative_pose(cam_params)
        else:
            c2w_poses = np.array([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
        c2w = torch.as_tensor(c2w_poses)[None]              # [1, n_frame, 4, 4]

        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(self.sample_n_frames, dtype=torch.bool, device=c2w.device)

        plucker_embedding = ray_condition(
            intrinsics, 
            c2w, 
            self.sample_size[0], 
            self.sample_size[1], 
            device='cpu',
            flip_flag=flip_flag
        )[0]
        plucker_embedding = plucker_embedding.permute(0, 3, 1, 2).contiguous()

        return pixel_values, condition_image, plucker_embedding, video_caption, flip_flag, clip_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                video, condition_image, plucker_embedding, \
                video_caption, flip_flag, clip_name = self.get_batch(idx)
                break
            except Exception as e:
                idx = rd.randint(0, self.length - 1)

        if self.use_flip:
            video = self.pixel_transforms[0](video)
            video = self.pixel_transforms[1](video, flip_flag)
            for transform in self.pixel_transforms[2:]:
                video = transform(video)
        else:
            for transform in self.pixel_transforms:
                video = transform(video)

        for transform in self.pixel_transforms:
            condition_image = transform(condition_image)

        sample = dict(pixel_values=video, condition_image=condition_image, plucker_embedding=plucker_embedding, video_caption=video_caption)
        if self.return_clip_name:
            sample['clip_name'] = clip_name
        return sample

