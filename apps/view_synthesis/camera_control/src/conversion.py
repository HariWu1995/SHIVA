import os
from tqdm import tqdm
from glob import glob 

import cv2


def convert_images_into_video(image_paths, output_path, fps: int = 10):

    # Read the first image to get dimensions
    init_path = image_paths[0]
    init_frame = cv2.imread(init_path)
    H, W, C = init_frame.shape

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 files
    video = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Write each frame
    for img_path in tqdm(image_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[Warning]: Skipping {img_path} (couldn't read)")
            continue
        resized = cv2.resize(frame, (W, H))
        video.write(resized)

    video.release()
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    image_folder = "./temp/cam_pov/img2trajvid/custom_traj/samples-rgb"
    image_paths = sorted([p for p in glob(image_folder + '/*.png')])

    output_path = image_folder + '.mp4'
    convert_images_into_video(image_paths, output_path)

