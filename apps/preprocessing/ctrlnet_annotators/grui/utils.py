import gradio as gr

from ..src import (
    apply_binary,
    apply_segment,
    apply_depth,
    apply_edge,
    apply_face,
    apply_pose,
)


def run_inference(

    # Inference
    func_name: str,
    image,

    # Basic
    config_basic_binary_threshold,

    # Depth
    config_depth_leres_thresh_fg,
    config_depth_leres_thresh_bg,
    config_depth_midas_thresh_bg,
    config_depth_midas_normalmap,
    config_depth_midas_z_scale,
    config_depth_dsine_iter,
    config_depth_dsine_fov,
    config_depth_dsine_res,

    # Edge
    config_edge_canny_thresh_low,
    config_edge_canny_thresh_high,
    config_edge_canny_structure,
    config_edge_canny_contrast,
    config_edge_canny_noisy,
    config_edge_pidi_threshold,
    config_edge_mlsd_thresh_score,
    config_edge_mlsd_thresh_dist,

    # Face
    config_face_mediapipe_max_faces,
    config_face_mediapipe_min_face_sz,
    config_face_mediapipe_threshold,
    
    # Pose
    config_pose_dense_cmap,
    config_pose_open_face,
    config_pose_open_body,
    config_pose_open_hand,
):
    func_name = func_name.lower()
    ftype, fname = func_name.split("_", maxsplit=1)

    config = dict()

    if ftype == "basic":
        if fname == "binary":
            return apply_binary(image, threshold=config_basic_binary_threshold)

    elif ftype == "depth":

        if fname in ["leres", "leres++"]:
            config.update(dict(
                thresh_a = config_depth_leres_thresh_fg,
                thresh_b = config_depth_leres_thresh_bg,
            ))

        elif fname in ["midas_v21", "midas_v21s", "dpt_large", "dpt_hybrid"]:
            config.update(dict(
                    z_value = config_depth_midas_z_scale,
                  bg_thresh = config_depth_midas_thresh_bg,
                use_normmap = config_depth_midas_normalmap,
            ))
        
        elif fname == "dsine":
            config.update(dict(
                   new_fov = config_depth_dsine_fov, 
                iterations = config_depth_dsine_iter, 
                resolution = config_depth_dsine_res,
            ))

        return apply_depth(image, model=fname, **config)

    elif ftype == "edge":

        if fname == "canny":
            config.update(dict(
                low_threshold = config_edge_canny_thresh_low,
                high_threshold = config_edge_canny_thresh_high, 
                    structure = config_edge_canny_structure,
                    contrast = config_edge_canny_contrast,
                    is_noisy = config_edge_canny_noisy,
            ))

        elif fname == "pidinet":
            config.update(dict(
                threshold = config_edge_pidi_threshold,
            ))

        elif fname in ["mlsd_large", "mlsd_tiny"]:
            config.update(dict(
                score_thresh = config_edge_mlsd_thresh_score,
                 dist_thresh = config_edge_mlsd_thresh_dist,
            ))

        return apply_edge(image, model=fname, **config)

    elif ftype == "face":

        if fname == "mediapipe":
            config.update(dict(
                max_faces            = config_face_mediapipe_max_faces,
                min_face_size_pixels = config_face_mediapipe_min_face_sz,
                min_confidence       = config_face_mediapipe_threshold,
            ))

        return apply_face(image, model=fname, **config)

    elif ftype == "pose":

        if fname == "densepose":
            config.update(dict(
                cmap = config_pose_dense_cmap,
            ))

        elif fname == "openpose":
            config.update(dict(
                include_body = config_pose_open_body,
                include_face = config_pose_open_face,
                include_hand = config_pose_open_hand,
            ))

        return apply_pose(image, model=fname, **config)

    elif ftype == "segment":
        return apply_segment(image, model=fname, **config)

    raise gradio.Error(f"💥 {func_name} is not supported!", duration=5)


