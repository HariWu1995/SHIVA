import os

import numpy as np
import cv2
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class VideoReader(Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")
        self.frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_count += 1
        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.transform = transform

    def __len__(self):
        return self.frame_count

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of range.")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            print(f"Failed to read frame at index {idx}.")
            return self[idx-1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


class VideoWriter:

    def __init__(self, path, frame_rate, frame_size, is_color=True):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'H264' if needed and supported
        writer = cv2.VideoWriter(path, fourcc, frame_rate, frame_size, isColor=is_color)
        self.path = path
        self.writer = writer

    def write(self, frames):
        # frames: [T, C, H, W], assumed to be torch.Tensor in [0,1] range
        frames = frames.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()  # [T, H, W, C]
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(frame_bgr)

    def close(self):
        self.writer.release()


class ImgSeqReader(Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.files = sorted(os.listdir(path))
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        with Image.open(os.path.join(self.path, self.files[idx])) as img:
            img.load()
        if self.transform is not None:
            return self.transform(img)
        return img


class ImgSeqWriter:

    def __init__(self, path, extension='jpg'):
        self.path = path
        self.extension = extension
        self.counter = 0
        os.makedirs(path, exist_ok=True)
    
    def write(self, frames):
        # frames: [T, C, H, W]
        for t in range(frames.shape[0]):
            F.to_pil_image(frames[t]).save(
                os.path.join(self.path, f'{self.counter:04d}.{self.extension}')
            )
            self.counter += 1
            
    def close(self):
        pass
        
