from PIL import Image

import numpy as np
import torch

from .data import get_parser
from .eval import compute_relative_inds, infer_prior_stats, infer_prior_inds
from .geometry import (
    generate_spiral_path, 
    generate_interp_path,
    get_arc_horizontal_w2cs, 
    get_default_intrinsics, 
    get_preset_pose_fov, 
    get_lookat,
)


def parse_task(
    task,
    scene,
    num_inputs,
    T,
    version_dict,
):
    anchor_indices = None
    anchor_c2ws = None
    anchor_Ks = None

    if task == "img2trajvid_s-prob":
        if num_inputs != 1:
            print("[⚠️] Task `img2trajvid_s-prob` only support 1-view conditioning. `num_inputs` will be overwritten as 1.")
        images_path, num_inputs, num_targets, \
                input_indices, anchor_indices, \
            c2ws, Ks, anchor_c2ws, anchor_Ks = run_task_img2trajvid_preset(scene, T, version_dict)

    else:
        parser = get_parser(parser_type="reconfusion", data_dir=scene, normalize=False)

        images_path = parser.image_paths
        camera_ids = parser.camera_ids

        Ks = np.concatenate([
                parser.Ks_dict[cam_id][None] for cam_id in camera_ids], axis=0)
        c2ws = parser.camtoworlds

        if num_inputs is None:
            assert len(parser.splits_per_num_input_frames.keys()) == 1
            num_inputs = list(parser.splits_per_num_input_frames.keys())[0]
            split_dict = parser.splits_per_num_input_frames[num_inputs]     # type: ignore

        elif isinstance(num_inputs, str):
            split_dict = parser.splits_per_num_input_frames[num_inputs]     # type: ignore
            num_inputs = int(num_inputs.split("-")[0])                      # for example 1_from32

        else:
            split_dict = parser.splits_per_num_input_frames[num_inputs]     # type: ignore

        if task == "img2img":
            images_path, num_targets, \
        input_indices, anchor_indices, \
            c2ws, Ks, anchor_c2ws, anchor_Ks = run_task_img2img(scene, num_inputs, split_dict, version_dict, 
                                                                T, c2ws, Ks, images_path)

        elif task == "img2vid":
            input_indices, anchor_indices, \
            num_targets, anchor_c2ws, anchor_Ks = run_task_img2vid(num_inputs, split_dict, version_dict,
                                                                    T, c2ws, Ks, images_path)

        elif task == "img2trajvid":
            images_path, num_targets, \
        input_indices, anchor_indices, \
            c2ws, Ks, anchor_c2ws, anchor_Ks, = run_task_img2trajvid(num_inputs, split_dict, version_dict,
                                                                    T, c2ws, Ks, images_path)

        else:
            raise ValueError(f"Task `{task}` is not supported!")
    
    return (
        images_path,
        num_inputs,
        num_targets,
        input_indices,
        anchor_indices,
        torch.tensor(c2ws[:, :3]).float(),
        torch.tensor(Ks).float(),
        torch.tensor(anchor_c2ws[:, :3]).float() if anchor_c2ws is not None else None,
        torch.tensor(anchor_Ks).float() if anchor_Ks is not None else None,
    )


def run_task_img2trajvid_preset(
    scene,
    T,
    version_dict,
):
    options = version_dict["options"]
    num_targets = options.get("num_targets", T - 1)
    num_inputs = 1
    num_anchors = infer_prior_stats(T, num_inputs, 
                    num_total_frames = num_targets, version_dict = version_dict)

    input_indices = [0]
    anchor_indices = np.linspace(1, num_targets, num_anchors).tolist()

    images_path = [scene] + [None] * num_targets

    c2ws, fovs = get_preset_pose_fov(
        option=options.get("traj_prior", "orbit"),
        num_frames=num_targets + 1,
        start_w2c=torch.eye(4),
        look_at=torch.Tensor([0, 0, 10]),
    )

    with Image.open(scene) as img:
        W, H = img.size
        aspect_ratio = W / H

    Ks = get_default_intrinsics(fovs, aspect_ratio=aspect_ratio)                    # unormalized
    Ks[:, :2] *= torch.tensor([W, H]).reshape(1, -1, 1).repeat(Ks.shape[0], 1, 1)   # normalized
    Ks = Ks.numpy()

    anchor_c2ws = c2ws[[round(ind) for ind in anchor_indices]]
    anchor_Ks = Ks[[round(ind) for ind in anchor_indices]]

    return (
        images_path, num_inputs, num_targets, 
        input_indices, anchor_indices, c2ws, Ks, anchor_c2ws, anchor_Ks
    )


def run_task_img2trajvid(
    num_inputs,
    split_dict,
    version_dict,
    T,
    c2ws,
    Ks,
    images_path,
):
    num_targets = len(split_dict["test_ids"])
    num_anchors = infer_prior_stats(T, num_inputs,
                    num_total_frames = num_targets, version_dict = version_dict)

    target_c2ws = c2ws[split_dict["test_ids"], :3]
    target_Ks   =   Ks[split_dict["test_ids"]]

    anchor_c2ws = target_c2ws[np.linspace(0, num_targets - 1, num_anchors).round().astype(np.int64)]
    anchor_Ks   =   target_Ks[np.linspace(0, num_targets - 1, num_anchors).round().astype(np.int64)]

    sampled_indices = split_dict["train_ids"] + split_dict["test_ids"]

    images_path = [images_path[i] for i in sampled_indices]
    c2ws = c2ws[sampled_indices]
    Ks = Ks[sampled_indices]

    input_indices = np.arange(num_inputs).tolist()
    anchor_indices = np.linspace(num_inputs, num_inputs + num_targets - 1, num_anchors).tolist()
    
    return (
        images_path, num_targets, 
        input_indices, anchor_indices, 
        c2ws, Ks, anchor_c2ws, anchor_Ks,
    )


def run_task_img2vid(
    num_inputs,
    split_dict,
    version_dict,
    T,
    c2ws,
    Ks,
    images_path,
):
    options = version_dict["options"]

    num_targets = len(images_path) - num_inputs
    num_anchors = infer_prior_stats(
        T,
        num_inputs,
        num_total_frames=num_targets,
        version_dict=version_dict,
    )

    input_indices = split_dict["train_ids"]
    anchor_indices = infer_prior_inds(c2ws, num_prior_frames=num_anchors,
                                        input_frame_indices=input_indices, options=options).tolist()
    num_anchors = len(anchor_indices)
    anchor_c2ws = c2ws[anchor_indices, :3]
    anchor_Ks = Ks[anchor_indices]

    return input_indices, anchor_indices, num_targets, anchor_c2ws, anchor_Ks


def run_task_img2img(
    scene,
    num_inputs,
    split_dict,
    version_dict,
    T,
    c2ws,
    Ks,
    images_path,
):
    # NOTE: 
    # in this setting, we should refrain from using all the other camera
    # info except ones from sampled_indices, and most importantly, the order.
    num_targets = len(split_dict["test_ids"])

    num_anchors = infer_prior_stats(T, num_inputs,
                    num_total_frames = num_targets, version_dict = version_dict)

    # we always sort all indices first
    sampled_indices = np.array(split_dict["train_ids"] + split_dict["test_ids"])
    sampled_indices = np.sort(sampled_indices)

    options = version_dict["options"]
    traj_prior = options.get("traj_prior", None)

    if traj_prior == "spiral":
        assert parser.bounds is not None
        anchor_c2ws = generate_spiral_path(
            c2ws[sampled_indices] @ np.diagflat([1, -1, -1, 1]),
            parser.bounds[sampled_indices],
            n_frames=num_anchors + 1,
            n_rots=2,
            zrate=0.5,
            endpoint=False,
        )[1:] @ np.diagflat([1, -1, -1, 1])

    elif traj_prior in ["interp","interpolated"]:
        assert num_inputs > 1
        anchor_c2ws = generate_interp_path(
            c2ws[split_dict["train_ids"], :3],
            round((num_anchors + 1) / (num_inputs - 1)),
            endpoint=False,
        )[1 : num_anchors + 1]

    elif traj_prior == "orbit":
        c2ws_th = torch.as_tensor(c2ws)
        lookat = get_lookat(c2ws_th[sampled_indices, :3, 3],
                            c2ws_th[sampled_indices, :3, 2])
        anchor_c2ws = torch.linalg.inv(
            get_arc_horizontal_w2cs(
                torch.linalg.inv(c2ws_th[split_dict["train_ids"][0]]),
                lookat,
                -F.normalize(c2ws_th[split_dict["train_ids"]][:, :3, 1].mean(0), dim=-1),
                num_frames=num_anchors + 1,
                endpoint=False,
            )
        ).numpy()[1:, :3]

    else:
        anchor_c2ws = None

    # anchor_Ks is default to be the first from target_Ks
    anchor_Ks = None

    images_path = [images_path[i] for i in sampled_indices]
    c2ws = c2ws[sampled_indices]
    Ks = Ks[sampled_indices]

    # absolute to relative indices
    input_indices = compute_relative_inds(sampled_indices, 
                                          np.array(split_dict["train_ids"]))

    anchor_indices = np.arange(sampled_indices.shape[0],
                               sampled_indices.shape[0] + num_anchors).tolist()
    return (
        images_path, num_targets, 
        input_indices, anchor_indices, 
        c2ws, Ks, anchor_c2ws, anchor_Ks,
    )

