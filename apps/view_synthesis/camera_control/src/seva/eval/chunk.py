import re
import math
import numpy as np

from collections import Counter

from .index import find_nearest_source_inds


def chunk_interp(
    chunk_strategy,
    gt_input_inds, task, T, 
    input_c2ws, test_c2ws,
    input_ords, test_ords, 
):
    M = input_c2ws.shape[0]
    N = test_c2ws.shape[0]

    # if chunk_strategy is `interp*`` and task is `img2trajvid*`,
    # we won't use input views because their order info within target views is unknown
    if "img2trajvid" in task:
        assert (list(range(len(gt_input_inds))) == gt_input_inds), \
                    "`img2trajvid` task should put `gt_input_inds` in start."
        input_c2ws =  input_c2ws[[ind for ind in range(M) if ind not in gt_input_inds]]
        input_ords = [input_ords[ind] for ind in range(M) if ind not in gt_input_inds]
        M = input_c2ws.shape[0]

    # HACK for test views
    input_ords = [0] + input_ords
    input_ords[-1] += 0.01  # ensuring last test stop is included

    # in the last forward when input_ords[-1] == test_ords[-1]
    test_ords  = np.array(test_ords)[None]
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
    chunks = []

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
    return chunks


def chunk_nearest(
    chunk_strategy,
    gt_input_inds,
    input_imgs, test_imgs,
    input_c2ws, test_c2ws,
    N, 
    T,
):
    chunks = []
    match = re.match(r"^nearest-(\d+)$", chunk_strategy)
    if match:
        nearest_num = int(match.group(1))
        assert (nearest_num < T), f"Nearest number of {nearest_num} should be less than {T}."
        
        # during the 2nd pass, translation is enough
        source_inds = find_nearest_source_inds(input_c2ws, 
                                                test_c2ws, nearest_num=nearest_num, mode="translation",)

        for i in range(0, N, T - nearest_num):
            i_nearest = i + T - nearest_num
            i_nearest_src = np.sort([
                ind for (ind, _) in Counter(source_inds[i:i_nearest].flatten().tolist()).most_common(nearest_num)
            ])
            chunk = input_imgs[i_nearest_src].tolist() \
                   + test_imgs[i : i_nearest].tolist()
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

            if len(chunk) == (T - len(prefix_inds)) or not candidate_input_inds:
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

        if chunk \
        and chunk != input_imgs[gt_input_inds].tolist():
            chunks.append(chunk + ["NULL"] * (T - len(chunk)))
    return chunks


def chunk_gt(
    chunk_strategy, 
    gt_input_inds, 
    N, T, options,
    test_c2ws,
):
    chunks = []
    num_test_seen = 0
    while num_test_seen < N:
        chunk = [f"!{i:03d}" for i in gt_input_inds]

        if chunk_strategy != "gt" \
        and num_test_seen > 0:
            pseudo_num_ratio = options.get("pseudo_num_ratio", 0.33)
            pseudo_num_max   = options.get("pseudo_num_max", 10_000)

            if (N - num_test_seen) >= \
                            math.floor((T - len(gt_input_inds)) * pseudo_num_ratio):
                pseudo_num = math.ceil((T - len(gt_input_inds)) * pseudo_num_ratio)
            else:
                pseudo_num = (T - len(gt_input_inds)) - (N - num_test_seen)
            pseudo_num = min(pseudo_num, pseudo_num_max)

            # Left-to-Right strategy
            if "ltr" in chunk_strategy:
                chunk.extend([
                    f"!{i + len(gt_input_inds):03d}"
                    for i in range(num_test_seen - pseudo_num, num_test_seen)
                ])

            # Nearest-source-index strategy
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
                # [🐦‍🔥 HACK] 
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
    return chunks

