from typing import List, Literal, Optional, Tuple, Union
from collections import Counter

from colorama import Fore, Style, init
init(autoreset=True)

import viser
import numpy as np
import torch

from .geometry import get_camera_dist, get_plucker_coordinates, to_hom, to_hom_pose


def get_wh_with_fixed_shortest_side(w, h, size):
    # size is smaller or equal to zero, we return original w h
    if size is None or size <= 0:
        return w, h
    if w < h:
        new_w = size
        new_h = int(size * h / w)
    else:
        new_h = size
        new_w = int(size * w / h)
    return new_w, new_h


def get_resizing_factor(
     target_shape: Tuple[int, int],  # H, W
    current_shape: Tuple[int, int],  # H, W
    cover_target: bool = True,
    # If True, the output shape will fully cover the target shape.
    # If False, the target shape will fully cover the output shape.
) -> float:
    r_bound  =  target_shape[1] /  target_shape[0]
    aspect_r = current_shape[1] / current_shape[0]
    if r_bound >= 1.0:
        if cover_target:
            if aspect_r >= r_bound:
                factor = min(target_shape) / min(current_shape)
            elif aspect_r < 1.0:
                factor = max(target_shape) / min(current_shape)
            else:
                factor = max(target_shape) / max(current_shape)
        else:
            if aspect_r >= r_bound:
                factor = max(target_shape) / max(current_shape)
            elif aspect_r < 1.0:
                factor = min(target_shape) / max(current_shape)
            else:
                factor = min(target_shape) / min(current_shape)
    else:
        if cover_target:
            if aspect_r <= r_bound:
                factor = min(target_shape) / min(current_shape)
            elif aspect_r > 1.0:
                factor = max(target_shape) / min(current_shape)
            else:
                factor = max(target_shape) / max(current_shape)
        else:
            if aspect_r <= r_bound:
                factor = max(target_shape) / max(current_shape)
            elif aspect_r > 1.0:
                factor = min(target_shape) / max(current_shape)
            else:
                factor = min(target_shape) / min(current_shape)
    return factor


def extend_dict(ds, d):
    for key in d.keys():
        if key in ds:
            ds[key] = torch.cat([ds[key], d[key]], 0)
        else:
            ds[key] = d[key]
    return ds


def replace_or_include_input_for_dict(
    samples,
    test_indices,
    imgs,
    c2w,
    K,
):
    samples_new = {}
    for sample, value in samples.items():
        if "rgb" in sample:
            imgs[test_indices] = (
                value[test_indices] if value.shape[0] == imgs.shape[0] else value
            ).to(device=imgs.device, dtype=imgs.dtype)
            samples_new[sample] = imgs
        elif "c2w" in sample:
            c2w[test_indices] = (
                value[test_indices] if value.shape[0] == c2w.shape[0] else value
            ).to(device=c2w.device, dtype=c2w.dtype)
            samples_new[sample] = c2w
        elif "intrinsics" in sample:
            K[test_indices] = (
                value[test_indices] if value.shape[0] == K.shape[0] else value
            ).to(device=K.device, dtype=K.dtype)
            samples_new[sample] = K
        else:
            samples_new[sample] = value
    return samples_new


def find_nearest_source_inds(
    source_c2ws,
    target_c2ws,
    nearest_num=1,
    mode="translation",
):
    dists = get_camera_dist(source_c2ws, target_c2ws, mode=mode).cpu().numpy()
    sorted_inds = np.argsort(dists, axis=0).T
    return sorted_inds[:, :nearest_num]


def chunk_input_and_test(
    T,
    input_c2ws,
    test_c2ws,
    input_ords,  # orders
    test_ords,  # orders
    options,
    task: str = "img2img",
    chunk_strategy: str = "gt",
    gt_input_inds: list = [],
):
    M, N = input_c2ws.shape[0], test_c2ws.shape[0]

    chunks = []
    if chunk_strategy.startswith("gt"):
        assert len(gt_input_inds) < T, \
            f"Number of gt input frames {len(gt_input_inds)} should be less than {T} when `gt` chunking strategy is used."
        assert list(range(M)) == gt_input_inds, \
            "All input_c2ws should be gt when `gt` chunking strategy is used."

        num_test_seen = 0
        while num_test_seen < N:
            chunk = [f"!{i:03d}" for i in gt_input_inds]

            if chunk_strategy != "gt" and num_test_seen > 0:
                pseudo_num_ratio = options.get("pseudo_num_ratio", 0.33)
                if (N - num_test_seen) >= math.floor((T - len(gt_input_inds)) * pseudo_num_ratio):
                    pseudo_num = math.ceil((T - len(gt_input_inds)) * pseudo_num_ratio)
                else:
                    pseudo_num = (T - len(gt_input_inds)) - (N - num_test_seen)
                pseudo_num = min(pseudo_num, options.get("pseudo_num_max", 10000))

                if "ltr" in chunk_strategy:
                    chunk.extend([
                        f"!{i + len(gt_input_inds):03d}"
                        for i in range(num_test_seen - pseudo_num, num_test_seen)
                    ])

                elif "nearest" in chunk_strategy:
                    source_inds = np.concatenate([
                        find_nearest_source_inds(test_c2ws[:num_test_seen],
                                                 test_c2ws[num_test_seen:], nearest_num=1, mode="rotation"),
                        find_nearest_source_inds(test_c2ws[:num_test_seen],
                                                 test_c2ws[num_test_seen:], nearest_num=1, mode="translation"),
                    ], axis=1)

                    ####### [HACK] keep running until pseudo num is stablized ########
                    temp_pseudo_num = pseudo_num
                    while True:
                        nearest_source_inds = np.concatenate([
                            np.sort(
                                [
                                    i for (i, _) in Counter([
                                                        item 
                                                    for item in source_inds[: T - len(gt_input_inds) - temp_pseudo_num].flatten().tolist()
                                                     if item != (num_test_seen - 1)
                                                    ]).most_common(pseudo_num - 1)
                                ],
                            ).astype(int),
                            [num_test_seen - 1],  # always keep the last one
                        ])

                        if len(nearest_source_inds) >= temp_pseudo_num:
                            break  # stablized
                        else:
                            temp_pseudo_num = len(nearest_source_inds)

                    pseudo_num = len(nearest_source_inds)
                    ########################################################################

                    chunk.extend(
                        [f"!{i + len(gt_input_inds):03d}" for i in nearest_source_inds]
                    )

                else:
                    raise NotImplementedError(
                        f"Chunking strategy {chunk_strategy} for the first pass is not implemented."
                    )

                chunk.extend([
                    f">{i:03d}"
                    for i in range(num_test_seen, min(num_test_seen + T - len(gt_input_inds) - pseudo_num, N))
                ])

            else:
                chunk.extend([
                    f">{i:03d}"
                    for i in range(num_test_seen, min(num_test_seen + T - len(gt_input_inds), N))
                ])

            num_test_seen += sum([1 for c in chunk if c.startswith(">")])
            if len(chunk) < T:
                chunk.extend(["NULL"] * (T - len(chunk)))
            chunks.append(chunk)

    elif chunk_strategy.startswith("nearest"):
        input_imgs = np.array([f"!{i:03d}" for i in range(M)])
        test_imgs  = np.array([f">{i:03d}" for i in range(N)])

        match = re.match(r"^nearest-(\d+)$", chunk_strategy)
        if match:
            nearest_num = int(match.group(1))
            assert nearest_num < T, \
                f"Nearest number of {nearest_num} should be less than {T}."

            source_inds = find_nearest_source_inds(
                input_c2ws,
                test_c2ws,
                nearest_num=nearest_num,
                mode="translation",  # during the second pass, consider translation only is enough
            )

            for i in range(0, N, T - nearest_num):
                nearest_source_inds = np.sort([
                    i for (i, _) in Counter(source_inds[i : i + T - nearest_num].flatten().tolist()).most_common(nearest_num)
                ])
                chunk = (
                     input_imgs[nearest_source_inds].tolist()
                    + test_imgs[i : i + T - nearest_num].tolist()
                )
                chunks.append(chunk + ["NULL"] * (T - len(chunk)))

        else:
            # do not always condition on gt cond frames
            if "gt" not in chunk_strategy:
                gt_input_inds = []

            source_inds = find_nearest_source_inds(
                input_c2ws,
                test_c2ws,
                nearest_num=1,
                mode="translation",  # during the second pass, consider translation only is enough
            )[:, 0]

            test_inds_per_input = {}
            for test_idx, input_idx in enumerate(source_inds):
                if input_idx not in test_inds_per_input:
                    test_inds_per_input[input_idx] = []
                test_inds_per_input[input_idx].append(test_idx)

            num_test_seen = 0
            chunk = input_imgs[gt_input_inds].tolist()
            candidate_input_inds = sorted(list(test_inds_per_input.keys()))

            while num_test_seen < N:
                input_idx = candidate_input_inds[0]
                test_inds = test_inds_per_input[input_idx]
                input_is_cond = input_idx in gt_input_inds
                prefix_inds = [] if input_is_cond else [input_idx]

                if len(chunk) == T - len(prefix_inds) or not candidate_input_inds:
                    if chunk:
                        chunk += ["NULL"] * (T - len(chunk))
                        chunks.append(chunk)
                        chunk = input_imgs[gt_input_inds].tolist()
                    if num_test_seen >= N:
                        break
                    continue

                candidate_chunk = (
                     input_imgs[prefix_inds].tolist() \
                    + test_imgs[  test_inds].tolist()
                )

                space_left = T - len(chunk)
                if len(candidate_chunk) <= space_left:
                    chunk.extend(candidate_chunk)
                    num_test_seen += len(test_inds)
                    candidate_input_inds.pop(0)
                else:
                    chunk.extend(candidate_chunk[:space_left])
                    num_input_idx = 0 if input_is_cond else 1
                    num_test_seen += space_left - num_input_idx
                    test_inds_per_input[input_idx] = test_inds[space_left - num_input_idx :]

                if len(chunk) == T:
                    chunks.append(chunk)
                    chunk = input_imgs[gt_input_inds].tolist()

            if chunk \
            and chunk != input_imgs[gt_input_inds].tolist():
                chunks.append(chunk + ["NULL"] * (T - len(chunk)))

    elif chunk_strategy.startswith("interp"):
        # `interp` chunk requires ordering info
        assert input_ords is not None and test_ords is not None, \
            "When using `interp` chunking strategy, ordering of input and test frames should be provided."

        # if chunk_strategy is `interp*`` and task is `img2trajvid*`, we will not
        # use input views since their order info within target views is unknown
        if "img2trajvid" in task:
            assert list(range(len(gt_input_inds))) == gt_input_inds, \
                "`img2trajvid` task should put `gt_input_inds` in start."
            input_c2ws = input_c2ws[[ind for ind in range(M) if ind not in gt_input_inds]]
            input_ords = [input_ords[ind] for ind in range(M) if ind not in gt_input_inds]
            M = input_c2ws.shape[0]

        input_ords = [0] + input_ords  # this is a  hack accounting for test views

        # before the first input view
        input_ords[-1] += 0.01  # this is a hack ensuring last test stop is included
        
        # in the last forward when input_ords[-1] == test_ords[-1]
        input_ords = np.array(input_ords)[:, None]
        input_ords_ = np.concatenate([input_ords[1:], np.full((1, 1), np.inf)])
        test_ords = np.array(test_ords)[None]

        in_stop_ranges = np.logical_and(
            np.repeat(input_ords, N, axis=1) <= np.repeat(test_ords, M + 1, axis=0),
            np.repeat(input_ords_, N, axis=1) > np.repeat(test_ords, M + 1, axis=0),
        )  # (M, N)
    
        assert (in_stop_ranges.sum(1) <= T - 2).all(), (
            "More anchor frames need to be sampled during the first pass to ensure "
            f"#target frames during each forward in the second pass will not exceed {T - 2}."
        )

        if input_ords[1, 0] <= test_ords[0, 0]:
            assert not in_stop_ranges[0].any()

        if input_ords[-1, 0] >= test_ords[0, -1]:
            assert not in_stop_ranges[-1].any()

        gt_chunk = [f"!{i:03d}" for i in gt_input_inds] if "gt" in chunk_strategy else []
        chunk = gt_chunk + []

        # any test views before the first input views
        if in_stop_ranges[0].any():
            for j, in_range in enumerate(in_stop_ranges[0]):
                if in_range:
                    chunk.append(f">{j:03d}")
        in_stop_ranges = in_stop_ranges[1:]

        i = 0
        base_i = len(gt_input_inds) if "img2trajvid" in task else 0
        chunk.append(f"!{i + base_i:03d}")
        while i < len(in_stop_ranges):
            in_stop_range = in_stop_ranges[i]
            if not in_stop_range.any():
                i += 1
                continue

            input_left = i + 1 < M
            space_left = T - len(chunk)
            if sum(in_stop_range) + input_left <= space_left:
                for j, in_range in enumerate(in_stop_range):
                    if in_range:
                        chunk.append(f">{j:03d}")
                i += 1
                if input_left:
                    chunk.append(f"!{i + base_i:03d}")

            else:
                chunk += ["NULL"] * space_left
                chunks.append(chunk)
                chunk = gt_chunk + [f"!{i + base_i:03d}"]

        if len(chunk) > 1:
            chunk += ["NULL"] * (T - len(chunk))
            chunks.append(chunk)

    else:
        raise NotImplementedError

    input_inds_per_chunk, test_inds_per_chunk = [], []
    input_sels_per_chunk, test_sels_per_chunk = [], []

    for chunk in chunks:
        input_inds = [int(img.removeprefix("!")) for img in chunk if img.startswith("!")]
        test_inds = [int(img.removeprefix(">")) for img in chunk if img.startswith(">")]
        input_sels = [chunk.index(img) for img in chunk if img.startswith("!")]
        test_sels = [chunk.index(img) for img in chunk if img.startswith(">")]
        input_inds_per_chunk.append(input_inds)
        input_sels_per_chunk.append(input_sels)
        test_inds_per_chunk.append(test_inds)
        test_sels_per_chunk.append(test_sels)

    if options.get("sampler_verbose", True):

        def colorize(item):
            if item.startswith("!"):
                return f"{Fore.RED}{item}{Style.RESET_ALL}"  # Red for items starting with '!'
            elif item.startswith(">"):
                return f"{Fore.GREEN}{item}{Style.RESET_ALL}"  # Green for items starting with '>'
            return item  # Default color if neither '!' nor '>'

        print("\nchunks:")
        for chunk in chunks:
            print(", ".join(colorize(item) for item in chunk))

    return (
        chunks,
        input_inds_per_chunk,  # ordering of input in raw sequence
        input_sels_per_chunk,  # ordering of input in one-forward sequence of length T
        test_inds_per_chunk,  # ordering of test in raw sequence
        test_sels_per_chunk,  # oredering of test in one-forward sequence of length T
    )


def is_k_in_dict(d, k):
    return any(map(lambda x: x.startswith(k), d.keys()))


def get_k_from_dict(d, k):
    media_d = {}
    for key, value in d.items():
        if key == k:
            return value
        if key.startswith(k):
            media = key.split("/")[-1]
            if media == "raw":
                return value
            media_d[media] = value
    if len(media_d) == 0:
        return torch.tensor([])
    assert len(media_d) == 1, f"multiple media found in {d} for key {k}: {media_d.keys()}"
    return media_d[media]


def update_kv_for_dict(d, k, v):
    for key in d.keys():
        if key.startswith(k):
            d[key] = v
    return d


def get_value_dict(
    curr_imgs,
    curr_imgs_clip,
    curr_input_frame_indices,
    curr_c2ws,
    curr_Ks,
    curr_input_camera_indices,
    all_c2ws,
    camera_scale,
):
    assert sorted(curr_input_camera_indices) == sorted(
        range(len(curr_input_camera_indices))
    )
    H = curr_imgs.shape[-2]
    W = curr_imgs.shape[-1]
    T = len(curr_imgs)
    F = 8

    value_dict = {}
    value_dict["cond_frames_without_noise"] = curr_imgs_clip[curr_input_frame_indices]
    value_dict["cond_frames"] = curr_imgs + 0.0 * torch.randn_like(curr_imgs)
    value_dict["cond_frames_mask"] = torch.zeros(T, dtype=torch.bool)
    value_dict["cond_frames_mask"][curr_input_frame_indices] = True
    value_dict["cond_aug"] = 0.0

    c2w = to_hom_pose(curr_c2ws.float())
    w2c = torch.linalg.inv(c2w)

    # camera centering
    ref_c2ws = all_c2ws
    camera_dist_2med = torch.norm(
        ref_c2ws[:, :3, 3] - ref_c2ws[:, :3, 3].median(0, keepdim=True).values,
        dim=-1,
    )
    valid_mask = camera_dist_2med <= torch.clamp(
        torch.quantile(camera_dist_2med, 0.97) * 10,
        max=1e6,
    )
    c2w[:, :3, 3] -= ref_c2ws[valid_mask, :3, 3].mean(0, keepdim=True)
    w2c = torch.linalg.inv(c2w)

    # camera normalization
    camera_dists = c2w[:, :3, 3].clone()
    translation_scaling_factor = (
              camera_scale if torch.isclose(torch.norm(camera_dists[0]), torch.zeros(1), atol=1e-5).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    w2c[:, :3, 3] *= translation_scaling_factor
    c2w[:, :3, 3] *= translation_scaling_factor

    value_dict["plucker_coordinate"] = get_plucker_coordinates(
        extrinsics_src=w2c[0],
        extrinsics=w2c,
        intrinsics=curr_Ks.float().clone(),
        target_size=(H // F, W // F),
    )

    value_dict["c2w"] = c2w
    value_dict["K"] = curr_Ks
    value_dict["camera_mask"] = torch.zeros(T, dtype=torch.bool)
    value_dict["camera_mask"][curr_input_camera_indices] = True

    return value_dict


def assemble(input, test,
            input_maps, test_maps):
    T = len(input_maps)
    assembled = torch.zeros_like(test[-1:]).repeat_interleave(T, dim=0)
    assembled[input_maps != -1] = input[input_maps[input_maps != -1]]
    assembled[ test_maps != -1] =  test[ test_maps[ test_maps != -1]]
    assert np.logical_xor(input_maps != -1, 
                           test_maps != -1).all()
    return assembled


def set_bg_color(server: viser.ViserServer | viser.ClientHandle, color = [39, 39, 42]):
    server.scene.set_background_image(np.array([[color]], dtype=np.uint8))

