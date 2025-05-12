import numpy as np

from ..geometry import get_camera_dist


def infer_prior_stats(
    T,
    num_input_frames,
    num_total_frames,
    version_dict,
):
    options = version_dict["options"]
    chunk_strategy = options.get("chunk_strategy", "nearest")

    T_first_pass = T[0] if isinstance(T, (list, tuple)) else T
    T_second_pass = T[1] if isinstance(T, (list, tuple)) else T

    # get traj_prior_c2ws for 2-pass sampling
    if chunk_strategy.startswith("interp"):
        # Start and end have alreay taken up two slots
        # +1 means we need X + 1 prior frames to bound X times forwards for all test frames

        # Tuning up `num_prior_frames_ratio` is helpful when you observe sudden jump in the
        # generated frames due to insufficient prior frames. This option is effective for
        # complicated trajectory and when `interp` strategy is used (usually semi-dense-view
        # regime). Recommended range is [1.0 (default), 1.5].
        num_prior_frames_ratio = options.get("num_prior_frames_ratio", 1.0)
        if num_input_frames >= options.get("num_input_semi_dense", 9):
            num_prior_frames = math.ceil(
            num_total_frames / (T_second_pass - 2) * num_prior_frames_ratio) + 1

            if num_prior_frames < T_first_pass - num_input_frames:
                num_prior_frames = T_first_pass - num_input_frames
            num_prior_frames = max(num_prior_frames, 
                        options.get("num_prior_frames", 0))

            T_first_pass = num_prior_frames + num_input_frames

            if "gt" in chunk_strategy:
                T_second_pass = T_second_pass + num_input_frames

            # Dynamically update context window length.
            version_dict["T"] = [T_first_pass, T_second_pass]

        else:
            num_prior_frames = math.ceil(
            num_total_frames / (T_second_pass - 2 - (
            num_input_frames if "gt" in chunk_strategy else 0)) * num_prior_frames_ratio) + 1

            if num_prior_frames < T_first_pass - num_input_frames:
                num_prior_frames = T_first_pass - num_input_frames
            num_prior_frames = max(num_prior_frames,
                        options.get("num_prior_frames", 0))
    else:
        num_prior_frames = max(T_first_pass - num_input_frames,
                                options.get("num_prior_frames", 0))
        if num_input_frames >= options.get("num_input_semi_dense", 9):
            T_first_pass = num_prior_frames + num_input_frames

            # Dynamically update context window length.
            version_dict["T"] = [T_first_pass, T_second_pass]

    return num_prior_frames


def infer_prior_inds(
    c2ws,
    num_prior_frames,
    input_frame_indices,
    options,
):
    chunk_strategy = options.get("chunk_strategy", "nearest")
    if chunk_strategy.startswith("interp"):
        prior_frame_indices = np.array([
            i for i in range(c2ws.shape[0]) 
               if i not in input_frame_indices
        ])
        prior_frame_indices = prior_frame_indices[
            np.ceil(
                np.linspace(0, prior_frame_indices.shape[0] - 1, num_prior_frames, endpoint=True)
            ).astype(int)
        ]  # having a ceil here is actually safer for corner case

    else:
        prior_frame_indices = []
        while len(prior_frame_indices) < num_prior_frames:
            closest_distance = np.abs(
                np.arange(c2ws.shape[0])[None] -
                np.concatenate([np.array(input_frame_indices), 
                                np.array(prior_frame_indices)])[:, None]
            ).min(0)
            prior_frame_indices.append(np.argsort(closest_distance)[-1])

    return np.sort(prior_frame_indices)


def compute_relative_inds(source_inds, target_inds):
    assert len(source_inds) > 2

    # compute relative indices of target_inds within source_inds
    relative_inds = []
    for ind in target_inds:
        if ind in source_inds:
            relative_ind = int(np.where(source_inds == ind)[0][0])
        elif ind < source_inds[0]:
            # extrapolate
            relative_ind = -((source_inds[0] - ind) / (source_inds[1] - source_inds[0]))
        elif ind > source_inds[-1]:
            # extrapolate
            relative_ind = len(source_inds) + (
                        (ind - source_inds[-1]) / (source_inds[-1] - source_inds[-2]))
        else:
            # interpolate
            lower_inds = source_inds[source_inds < ind]
            upper_inds = source_inds[source_inds > ind]
            if  len(lower_inds) > 0 \
            and len(upper_inds) > 0:
                lower_ind = lower_inds[-1]
                upper_ind = upper_inds[0]
                rel_lower_ind = int(np.where(source_inds == lower_ind)[0][0])
                rel_upper_ind = int(np.where(source_inds == upper_ind)[0][0])
                relative_ind = rel_lower_ind + (
                             ind - lower_ind) / (
                       upper_ind - lower_ind) * (rel_upper_ind - rel_lower_ind)
            else:
                # Out of range
                relative_inds.append(float("nan"))  # Or some other placeholder
        relative_inds.append(relative_ind)
    return relative_inds


def find_nearest_source_inds(
    source_c2ws,
    target_c2ws,
    nearest_num=1,
    mode="translation",
):
    dists = get_camera_dist(source_c2ws, target_c2ws, mode=mode).cpu().numpy()
    sorted_inds = np.argsort(dists, axis=0).T
    return sorted_inds[:, :nearest_num]

