import gradio as gr
import numpy as np


def load_arguments():

    # Basic
    config_basic_binary_threshold = gr.Slider(label="Threshold", minimum=-1, maximum=255, step=1, value=-1)

    # Depth
    config_depth_leres_thresh_fg = gr.Slider(label="Foreground Threshold", value=0, minimum=0, maximum=255, step=1)
    config_depth_leres_thresh_bg = gr.Slider(label="Background Threshold", value=0, minimum=0, maximum=255, step=1)
    
    config_depth_midas_thresh_bg = gr.Slider(label="Background Threshold", value=.1, minimum=0., maximum=1., step=.1)
    config_depth_midas_normalmap = gr.Checkbox(label="Return NormalMap", value=False)
    config_depth_midas_z_scale   = gr.Number(label="Z-scale (NormalMap)", value=np.pi*2, minimum=0, maximum=255, step=.1)

    config_depth_dsine_fov = gr.Number(label="Field-of-View", value=60., minimum=0, maximum=255, step=.1)
    config_depth_dsine_iter = gr.Slider(label="#Iterations", value=5, minimum=0, maximum=100, step=1)
    config_depth_dsine_res = gr.Slider(label="Resolution", value=512, minimum=8, maximum=2048, step=4)

    # Edge
    config_edge_canny_thresh_low = gr.Slider(label="Low Threshold", value=-1, minimum=-1, maximum=255, step=1)
    config_edge_canny_thresh_high = gr.Slider(label="High Threshold", value=-1, minimum=-1, maximum=255, step=1)
    config_edge_canny_structure = gr.Dropdown(label="Structure", value="normal", choices=["normal","clean","smooth"])
    config_edge_canny_contrast = gr.Dropdown(label="Contrast", value="normal", choices=["normal","high","low"])
    config_edge_canny_noisy = gr.Checkbox(label="Noisy", value=False)

    config_edge_pidi_threshold = gr.Slider(label="Threshold", value=0., minimum=0., maximum=1., step=.1)

    config_edge_mlsd_thresh_score = gr.Slider(label="Confidence Threshold", value=.1, minimum=0., maximum=1., step=.1)
    config_edge_mlsd_thresh_dist = gr.Slider(label="Distance Threshold", value=20., minimum=0., maximum=250., step=1.)

    # Face
    config_face_mediapipe_max_faces = gr.Number(label="Max #faces", value=10, minimum=1, maximum=100, step=1)
    config_face_mediapipe_min_face_sz = gr.Number(label="Min face size (w*h)", value=-1, minimum=-1, maximum=100_00, step=10)
    config_face_mediapipe_threshold = gr.Slider(label="Confidence Threshold", value=.1, minimum=0., maximum=1., step=.1)

    # Pose
    config_pose_dense_cmap = gr.Dropdown(label="ColorMap (Viz)", value="viridis", choices=["viridis","parula"])

    config_pose_open_face = gr.Checkbox(label="Face Included", value=True)
    config_pose_open_body = gr.Checkbox(label="Body Included", value=True)
    config_pose_open_hand = gr.Checkbox(label="Hand Included", value=True)

    # Hide all optional inputs by default
    config_basic_binary_threshold.visible = True
    config_depth_leres_thresh_fg.visible = False
    config_depth_leres_thresh_bg.visible = False
    config_depth_midas_thresh_bg.visible = False
    config_depth_midas_normalmap.visible = False
    config_depth_midas_z_scale.visible = False
    config_depth_dsine_iter.visible = False
    config_depth_dsine_fov.visible = False
    config_depth_dsine_res.visible = False
    config_edge_canny_thresh_low.visible = False
    config_edge_canny_thresh_high.visible = False
    config_edge_canny_noisy.visible = False
    config_edge_canny_structure.visible = False
    config_edge_canny_contrast.visible = False
    config_edge_pidi_threshold.visible = False
    config_edge_mlsd_thresh_score.visible = False
    config_edge_mlsd_thresh_dist.visible = False
    config_face_mediapipe_max_faces.visible = False
    config_face_mediapipe_min_face_sz.visible = False
    config_face_mediapipe_threshold.visible = False
    config_pose_dense_cmap.visible = False
    config_pose_open_face.visible = False
    config_pose_open_body.visible = False
    config_pose_open_hand.visible = False

    return [
        # Basic
        config_basic_binary_threshold,

        # Depth
        config_depth_leres_thresh_fg,
        config_depth_leres_thresh_bg,
        config_depth_midas_thresh_bg,
        config_depth_midas_z_scale,
        config_depth_midas_normalmap,
        config_depth_dsine_iter,
        config_depth_dsine_fov,
        config_depth_dsine_res,

        # Edge
        config_edge_canny_thresh_low,
        config_edge_canny_thresh_high,
        config_edge_canny_noisy,
        config_edge_canny_structure,
        config_edge_canny_contrast,
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
    ]


def update_arguments(

    func_name: str,

    # Basic
    config_basic_binary_threshold,

    # Depth
    config_depth_leres_thresh_fg,
    config_depth_leres_thresh_bg,
    config_depth_midas_thresh_bg,
    config_depth_midas_z_scale,
    config_depth_midas_normalmap,
    config_depth_dsine_iter,
    config_depth_dsine_fov,
    config_depth_dsine_res,

    # Edge
    config_edge_canny_thresh_low,
    config_edge_canny_thresh_high,
    config_edge_canny_noisy,
    config_edge_canny_structure,
    config_edge_canny_contrast,
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

    return (
        # Basic - Binary
        gr.update(visible=(ftype == "basic" and fname == "binary")),

        # Depth - LeRes / LeRes++
        gr.update(visible=(ftype == "depth" and fname in ["leres", "leres++"])),
        gr.update(visible=(ftype == "depth" and fname in ["leres", "leres++"])),

        # Depth - MiDas / DPT
        gr.update(visible=(ftype == "depth" and fname in ["dpt_large", "dpt_hybrid", "midas_v21", "midas_v21s"])),
        gr.update(visible=(ftype == "depth" and fname in ["dpt_large", "dpt_hybrid", "midas_v21", "midas_v21s"])),
        gr.update(visible=(ftype == "depth" and fname in ["dpt_large", "dpt_hybrid", "midas_v21", "midas_v21s"])),

        # Depth - DSINE
        gr.update(visible=(ftype == "depth" and fname == "dsine")),
        gr.update(visible=(ftype == "depth" and fname == "dsine")),
        gr.update(visible=(ftype == "depth" and fname == "dsine")),

        # Edge - Canny
        gr.update(visible=(ftype == "edge" and fname == "canny")),
        gr.update(visible=(ftype == "edge" and fname == "canny")),
        gr.update(visible=(ftype == "edge" and fname == "canny")),
        gr.update(visible=(ftype == "edge" and fname == "canny")),
        gr.update(visible=(ftype == "edge" and fname == "canny")),

        # Edge - PiDiNet
        gr.update(visible=(ftype == "edge" and fname == "pidinet")),

        # Edge - MSLD
        gr.update(visible=(ftype == "edge" and fname in ["mlsd_large", "mlsd_tiny"])),
        gr.update(visible=(ftype == "edge" and fname in ["mlsd_large", "mlsd_tiny"])),

        # Face - MediaPipe
        gr.update(visible=(ftype == "face" and fname == "mediapipe")),
        gr.update(visible=(ftype == "face" and fname == "mediapipe")),
        gr.update(visible=(ftype == "face" and fname == "mediapipe")),
        
        # Pose - DensePose
        gr.update(visible=(ftype == "pose" and fname == "densepose")),

        # Pose - OpenPose
        gr.update(visible=(ftype == "pose" and fname == "openpose")),
        gr.update(visible=(ftype == "pose" and fname == "openpose")),
        gr.update(visible=(ftype == "pose" and fname == "openpose")),
    )
