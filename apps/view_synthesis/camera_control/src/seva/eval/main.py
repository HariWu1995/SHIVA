import re
from typing import List, Literal, Optional, Tuple, Union
from collections import Counter

import json
import math
import numpy as np

import torch
try:
    # Check if version string contains 'dev' or 'nightly'
    version = torch.__version__
    IS_TORCH_NIGHTLY = "dev" in version
    if IS_TORCH_NIGHTLY:
        torch._dynamo.config.cache_size_limit = 128                 # type: ignore[assignment]
        torch._dynamo.config.accumulated_cache_size_limit = 1024    # type: ignore[assignment]
        torch._dynamo.config.force_parameter_static_shapes = False  # type: ignore[assignment]

except Exception:
    IS_TORCH_NIGHTLY = False


def pad_indices(
    input_indices: List[int],
    test_indices: List[int],
    T: int,
    padding_mode: Literal["first", "last", "none"] = "last",
):
    assert padding_mode in ["last", "none"], "`first` padding is not supported yet."
    if padding_mode == "last":
        padded_indices = [i for i in range(T) if i not in (input_indices + test_indices)]
    else:
        padded_indices = []

    input_selects = list(range(len(input_indices)))
    test_selects = list(range(len(test_indices)))

    if max(input_indices) > max(test_indices):
        # last elem from input
        input_selects += [input_selects[-1]] * len(padded_indices)
        input_indices = input_indices + padded_indices
        sorted_inds = np.argsort(input_indices)
        input_indices = [input_indices[ind] for ind in sorted_inds]
        input_selects = [input_selects[ind] for ind in sorted_inds]
    else:
        # last elem from test
        test_selects += [test_selects[-1]] * len(padded_indices)
        test_indices = test_indices + padded_indices
        sorted_inds = np.argsort(test_indices)
        test_indices = [test_indices[ind] for ind in sorted_inds]
        test_selects = [test_selects[ind] for ind in sorted_inds]

    if padding_mode == "last":
        input_maps = np.array([-1] * T)
        test_maps = np.array([-1] * T)
    else:
        input_maps = np.array([-1] * (len(input_indices) + len(test_indices)))
        test_maps = np.array([-1] * (len(input_indices) + len(test_indices)))

    input_maps[input_indices] = input_selects
    test_maps[test_indices] = test_selects
    return (
        input_indices, test_indices, 
        input_maps, test_maps
    )


def assemble(inputs, tests, input_maps, test_maps):
    T = len(input_maps)
    assembled = torch.zeros_like(test[-1:]).repeat_interleave(T, dim=0)
    assembled[input_maps != -1] = inputs[input_maps[input_maps != -1]]
    assembled[ test_maps != -1] =  tests[ test_maps[ test_maps != -1]]
    assert np.logical_xor(input_maps != -1, 
                           test_maps != -1).all()
    return assembled


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
    M = input_c2ws.shape[0]
    N = test_c2ws.shape[0]

    chunks = []
    if chunk_strategy.startswith("gt"):
        assert len(gt_input_inds) < T, (
            f"Number of gt input frames {len(gt_input_inds)} should be "
            f"less than {T} when `gt` chunking strategy is used."
        )
        assert (list(range(M)) == gt_input_inds), \
        "All input_c2ws should be gt when `gt` chunking strategy is used."

        num_test_seen = 0
        while num_test_seen < N:
            chunk = [f"!{i:03d}" for i in gt_input_inds]
            if chunk_strategy != "gt" and num_test_seen > 0:
                pseudo_num_ratio = options.get("pseudo_num_ratio", 0.33)
                if (N - num_test_seen) >= math.floor(
                   (T - len(gt_input_inds)) * pseudo_num_ratio
                ):
                    pseudo_num = math.ceil((T - len(gt_input_inds)) * pseudo_num_ratio)
                else:
                    pseudo_num = (T - len(gt_input_inds)) - (N - num_test_seen)
                pseudo_num = min(pseudo_num, options.get("pseudo_num_max", 10_000))

                if "ltr" in chunk_strategy:
                    chunk.extend([
                        f"!{i + len(gt_input_inds):03d}"
                        for i in range(num_test_seen - pseudo_num, num_test_seen)
                    ])

                elif "nearest" in chunk_strategy:
                    source_inds = np.concatenate([
                           find_nearest_source_inds(test_c2ws[:num_test_seen],
                                                    test_c2ws[num_test_seen:],
                                                    nearest_num=1,  # pseudo_num,
                                                    mode="rotation"),
                           find_nearest_source_inds(test_c2ws[:num_test_seen],
                                                    test_c2ws[num_test_seen:],
                                                    nearest_num=1,  # pseudo_num,
                                                    mode="translation"),
                    ], axis=1)

                    ############################################
                    # [HACK ALERT] 
                    # keep running until pseudo num is stablized
                    temp_pseudo_num = pseudo_num
                    while True:
                        nearest_source_inds = np.concatenate([
                            np.sort([
                                    ind
                                for (ind, _) in Counter([
                                    item 
                                for item in source_inds[: T - len(gt_input_inds) - temp_pseudo_num].flatten().tolist()
                                 if item != (num_test_seen - 1)  # exclude the last one here
                                ]).most_common(pseudo_num - 1)
                            ]).astype(int),
                            [num_test_seen - 1],  # always keep the last one
                        ])

                        if len(nearest_source_inds) >= temp_pseudo_num:
                            break  # stablized
                        else:
                            temp_pseudo_num = len(nearest_source_inds)
                    pseudo_num = len(nearest_source_inds)
                    ##################################################

                    chunk.extend([f"!{i + len(gt_input_inds):03d}" for i in nearest_source_inds])
                
                else:
                    raise NotImplementedError(
                        f"Chunking strategy {chunk_strategy} for the first pass is not implemented."
                    )

                chunk.extend([
                    f">{i:03d}"
                    for i in range(num_test_seen, 
                                min(num_test_seen + T - len(gt_input_inds) - pseudo_num, N))
                ])

            else:
                chunk.extend([
                    f">{i:03d}"
                    for i in range(num_test_seen,
                                min(num_test_seen + T - len(gt_input_inds), N))
                ])

            num_test_seen += sum([1 for c in chunk if c.startswith(">")])

            if len(chunk) < T:
                chunk.extend(["NULL"] * (T - len(chunk)))
            chunks.append(chunk)

    elif chunk_strategy.startswith("nearest"):
        input_imgs = np.array([f"!{i:03d}" for i in range(M)])
        test_imgs = np.array([f">{i:03d}" for i in range(N)])

        match = re.match(r"^nearest-(\d+)$", chunk_strategy)
        if match:
            nearest_num = int(match.group(1))
            assert (nearest_num < T), f"Nearest number of {nearest_num} should be less than {T}."
            
            source_inds = find_nearest_source_inds(
                input_c2ws,
                test_c2ws,
                nearest_num=nearest_num,
                mode="translation",  # during the second pass, consider translation only is enough
            )

            for i in range(0, N, T - nearest_num):
                nearest_source_inds = np.sort([
                    ind for (ind, _) in Counter(source_inds[i : i + T - nearest_num].flatten().tolist()).most_common(nearest_num)
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
                input_is_cond = input_idx in gt_input_inds

                test_inds = test_inds_per_input[input_idx]
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
                    input_imgs[prefix_inds].tolist() 
                    + test_imgs[test_inds].tolist()
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
                    test_inds_per_input[input_idx] = \
                    test_inds[space_left - num_input_idx :]

                if len(chunk) == T:
                    chunks.append(chunk)
                    chunk = input_imgs[gt_input_inds].tolist()

            if chunk and chunk != input_imgs[gt_input_inds].tolist():
                chunks.append(chunk + ["NULL"] * (T - len(chunk)))

    elif chunk_strategy.startswith("interp"):
        # `interp` chunk requires ordering info
        assert input_ords is not None and test_ords is not None, \
            "When using `interp` chunking strategy, ordering of input and test frames should be provided."

        # if chunk_strategy is `interp*`` and task is `img2trajvid*`, we will not
        # use input views since their order info within target views is unknown
        if "img2trajvid" in task:
            assert (list(range(len(gt_input_inds))) == gt_input_inds), "`img2trajvid` task should put `gt_input_inds` in start."
            input_c2ws =  input_c2ws[[ind for ind in range(M) if ind not in gt_input_inds]]
            input_ords = [input_ords[ind] for ind in range(M) if ind not in gt_input_inds]
            M = input_c2ws.shape[0]

        input_ords = [0] + input_ords  # this is a HACK accounting for test views
        input_ords[-1] += 0.01  # this is a HACK ensuring last test stop is included

        # in the last forward when input_ords[-1] == test_ords[-1]
        test_ords = np.array(test_ords)[None]
        input_ords = np.array(input_ords)[:, None]
        input_ords_ = np.concatenate([input_ords[1:], np.full((1, 1), np.inf)])

        in_stop_ranges = np.logical_and(
            np.repeat(input_ords , N, axis=1) <= np.repeat(test_ords, M + 1, axis=0),
            np.repeat(input_ords_, N, axis=1) >  np.repeat(test_ords, M + 1, axis=0),
        )  # (M, N)

        assert (in_stop_ranges.sum(1) <= T - 2).all(), (
            "More anchor frames need to be sampled during the first pass to ensure "
            f"target frames during each forward in the second pass will not exceed {T - 2}."
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

    input_inds_per_chunk = []
    input_sels_per_chunk = []
    test_inds_per_chunk = []
    test_sels_per_chunk = []
    for chunk in chunks:

        input_inds = []
        input_sels = []
        test_inds = []
        test_sels = []

        for img in chunk:

            if img.startswith("!"):
                input_inds.append(int(img.removeprefix("!")))
                input_sels.append(chunk.index(img))

            if img.startswith(">"):
                test_inds.append(int(img.removeprefix(">")))
                test_sels.append(chunk.index(img))

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


def create_transforms_simple(save_path, img_paths, img_whs, c2ws, Ks):
    import os.path as osp

    out_frames = []
    for img_path, img_wh, c2w, K in zip(img_paths, img_whs, c2ws, Ks):
        out_frame = {
            "fl_x": K[0][0].item(),
            "fl_y": K[1][1].item(),
            "cx": K[0][2].item(),
            "cy": K[1][2].item(),
            "w": img_wh[0].item(),
            "h": img_wh[1].item(),
            "file_path": f"./{osp.relpath(img_path, start=save_path)}" if img_path else None,
            "transform_matrix": c2w.tolist(),
        }
        out_frames.append(out_frame)
    out = {
        # "camera_model": "PINHOLE",
        "orientation_override": "none",
        "frames": out_frames,
    }
    with open(osp.join(save_path, "transforms.json"), "w") as of:
        json.dump(out, of, indent=5)

