import os
import os.path as osp

import copy
import json
import time

from datetime import datetime
from typing import Literal
from pathlib import Path

import torch
from einops import rearrange

import numpy as np
import imageio.v3 as iio

import viser
from viser import ViserServer
from viser import transforms as vTr

from .cam_traj import load_gui
from ..src.utils import set_bg_color
from ..src.scene import transform_img_and_K
from ..src.geometry import get_default_intrinsics, normalize_scene
from ..src.preprocessor import Dust3r


basic_trajectories = Literal[
    "orbit",
    "spiral",
    "lemniscate",
    "zoom-in",
    "zoom-out",
    "dolly zoom-in",
    "dolly zoom-out",
    "move-forward",
    "move-backward",
    "move-up",
    "move-down",
    "move-left",
    "move-right",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_transforms_simple(save_path, img_paths, img_whs, c2ws, Ks):
    out_frames = []
    for img_path, img_wh, c2w, K in zip(img_paths, img_whs, c2ws, Ks):
        out_frame = {
            "fl_x": K[0][0].item(),
            "fl_y": K[1][1].item(),
            "cx": K[0][2].item(),
            "cy": K[1][2].item(),
            "w": img_wh[0].item(),
            "h": img_wh[1].item(),
            "file_path": f"./{osp.relpath(img_path, start=save_path)}" if img_path is not None else None,
            "transform_matrix": c2w.tolist(),
        }
        out_frames.append(out_frame)
    out = {
        # "camera_model": "PINHOLE",
        "orientation_override": "none",
        "frames": out_frames,
    }
    with open(osp.join(save_path, "transforms.json"), "w") as of:
        json.dump(out, of, indent=4)


ROOT_DIR = Path(__file__).resolve().parents[4]

MODEL_DIR = ROOT_DIR / "checkpoints"
DUST3R_PATH = MODEL_DIR / "3d/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

class Renderer:

    def __init__(self, server: ViserServer, model_path: str = DUST3R_PATH):
        self.server = server
        self.DUST3R = Dust3r(device=DEVICE, model_path=model_path)
        self.gui_state = None

    def preprocess(
        self, 
        input_img_path_or_tuples: list[tuple[str, None]] | str,
        center_crop: bool = False,
    ) -> tuple[dict, dict, dict]:
        # FIXME: aspect ratio is always kept and shorter side is resized to 576.
        # This is only to make GUI option fewer though, changing it still works.
        # Has to be 64-multiple for the network.
        shorter: int = 576
        shorter = round(shorter / 64) * 64
        dtype = torch.float32

        # `Basic` mode
        # NOTE: just hardcode the camera parameters and ignore points.
        if isinstance(input_img_path_or_tuples, str):
            input_imgs = iio.imread(input_img_path_or_tuples)
            input_imgs = torch.as_tensor(input_imgs / 255.0, dtype=dtype)[None, ..., :3]
            input_imgs = transform_img_and_K(
                            input_imgs.permute(0, 3, 1, 2), shorter, K=None, size_stride=64,
                        )[0].permute(0, 2, 3, 1)
            input_Ks = get_default_intrinsics(aspect_ratio=input_imgs.shape[2] / input_imgs.shape[1])
            input_c2ws = torch.eye(4)[None]

            # Simulate a small time interval such that gradio can update propgress properly.
            time.sleep(0.1)
            return {
                "input_imgs": input_imgs,
                "input_Ks": input_Ks,
                "input_c2ws": input_c2ws,
                "input_wh": (input_imgs.shape[2], input_imgs.shape[1]),
                "points": [np.zeros((0, 3))],
                "point_colors": [np.zeros((0, 3))],
                "scene_scale": 1.0,
            }
        
        # `Advance` mode: use Dust3r to extract camera parameters and points.    
        else:
            img_paths = [p for (p, _) in input_img_path_or_tuples]
            (
                input_imgs,
                input_Ks,
                input_c2ws,
                points,
                point_colors,
            ) = self.DUST3R.infer_cameras_and_points(img_paths)

            num_inputs = len(img_paths)
            if num_inputs == 1:
                input_imgs = input_imgs[:1]
                input_Ks   = input_Ks  [:1]
                input_c2ws = input_c2ws[:1]
                points       = points[:1]
                point_colors = point_colors[:1]
            input_imgs = [img[..., :3] for img in input_imgs]

            # Normalize the scene.
            point_chunks = [p.shape[0] for p in points]
            point_indices = np.cumsum(point_chunks)[:-1]
            input_c2ws, points, _ = normalize_scene(  # type: ignore
                input_c2ws,
                np.concatenate(points, 0),
                camera_center_method="poses",
            )
            points = np.split(points, point_indices, 0)
            # Scale camera and points for viewport visualization.
            scene_scale = np.median(
                np.ptp(np.concatenate([input_c2ws[:, :3, 3], *points], 0), -1)
            )
            input_c2ws[:, :3, 3] /= scene_scale
            points = [point / scene_scale for point in points]
            input_imgs = [torch.as_tensor(img / 255.0, dtype=torch.float32) for img in input_imgs]
            input_Ks = torch.as_tensor(input_Ks)
            input_c2ws = torch.as_tensor(input_c2ws)

            new_input_imgs = []
            new_input_Ks = []
            for img, K in zip(input_imgs, input_Ks):
                img = rearrange(img, "h w c -> 1 c h w")
                if center_crop:
                    img, K = transform_img_and_K(img, (shorter, shorter), K=K[None])
                else:
                    img, K = transform_img_and_K(img, shorter, K=K[None], size_stride=64)
                assert isinstance(K, torch.Tensor)
                K = K / K.new_tensor([img.shape[-1], img.shape[-2], 1])[:, None]
                new_input_imgs.append(img)
                new_input_Ks.append(K)
    
            input_imgs = torch.cat(new_input_imgs, 0)
            input_imgs = rearrange(input_imgs, "b c h w -> b h w c")[..., :3]
            input_Ks = torch.cat(new_input_Ks, 0)
            return {
                "input_imgs": input_imgs,
                "input_Ks": input_Ks,
                "input_c2ws": input_c2ws,
                "input_wh": (input_imgs.shape[2], input_imgs.shape[1]),
                "points": points,
                "point_colors": point_colors,
                "scene_scale": scene_scale,
            }

    def visualize_scene(self, preprocessed: dict):

        server = self.server
        server.scene.reset()
        server.gui.reset()
        set_bg_color(server)

        input_imgs = preprocessed["input_imgs"]
        input_wh   = preprocessed["input_wh"]
        input_Ks   = preprocessed["input_Ks"]
        input_c2ws = preprocessed["input_c2ws"]
        scene_scale = preprocessed["scene_scale"]
        point_colors = preprocessed["point_colors"]
        points       = preprocessed["points"]
        
        W, H = input_wh

        server.scene.set_up_direction(-input_c2ws[..., :3, 1].mean(0).numpy())

        # Use first image as default fov.
        assert input_imgs[0].shape[:2] == (H, W)
        if H > W:
            init_fov = 2 * np.arctan(1 / (2 * input_Ks[0, 0, 0].item()))
        else:
            init_fov = 2 * np.arctan(1 / (2 * input_Ks[0, 1, 1].item()))
        init_fov_deg = float(init_fov / np.pi * 180.0)

        frustum_nodes, pcd_nodes = [], []
        for i in range(len(input_imgs)):
            K = input_Ks[i]
            frustum = server.scene.add_camera_frustum(
                f"/scene_assets/cameras/{i}",
                fov=2 * np.arctan(1 / (2 * K[1, 1].item())),
                aspect=W / H,
                scale=0.1 * scene_scale,
                image=(input_imgs[i].numpy() * 255.0).astype(np.uint8),
                wxyz=vTr.SO3.from_matrix(input_c2ws[i, :3, :3].numpy()).wxyz,
                position=input_c2ws[i, :3, 3].numpy(),
            )

            def get_handler(frustum):
                def handler(event: viser.GuiEvent) -> None:
                    assert event.client_id is not None
                    client = server.get_clients()[event.client_id]
                    with client.atomic():
                        client.camera.position = frustum.position
                        client.camera.wxyz = frustum.wxyz
                        # Set look_at as the projected origin onto the frustum's forward direction.
                        look_direction = vTr.SO3(frustum.wxyz).as_matrix()[:, 2]
                        position_origin = -frustum.position
                        client.camera.look_at = (
                            frustum.position
                            + np.dot(look_direction, position_origin)
                            / np.linalg.norm(position_origin)
                            * look_direction
                        )

                return handler

            frustum.on_click(get_handler(frustum))  # type: ignore
            frustum_nodes.append(frustum)

            pcd = server.scene.add_point_cloud(
                f"/scene_assets/points/{i}",
                points[i],
                point_colors[i],
                point_size=0.01 * scene_scale,
                point_shape="circle",
            )
            pcd_nodes.append(pcd)

        with server.gui.add_folder("Scene scale", expand_by_default=False, order=200):

            camera_scale_slider = server.gui.add_slider("Log camera scale", initial_value=0.0, min=-2.0, max=2.0, step=0.1)
            @camera_scale_slider.on_update
            def _(_) -> None:
                for i in range(len(frustum_nodes)):
                    frustum_nodes[i].scale = 0.1 * scene_scale * 10 ** camera_scale_slider.value

            point_scale_slider = server.gui.add_slider("Log point scale", initial_value=0.0, min=-2.0, max=2.0, step=0.1)
            @point_scale_slider.on_update
            def _(_) -> None:
                for i in range(len(pcd_nodes)):
                    pcd_nodes[i].point_size = (0.01 * scene_scale * 10 ** point_scale_slider.value)

        self.gui_state = load_gui(
            server,
            init_fov=init_fov_deg,
            img_wh=input_wh,
            scene_scale=scene_scale,
        )

    def get_target_c2ws_and_Ks_from_gui(self, preprocessed: dict):
        input_wh = preprocessed["input_wh"]
        W, H = input_wh
        
        gui_state = self.gui_state
        assert gui_state is not None
        assert gui_state.camera_traj_list is not None
        
        target_c2ws, target_Ks = [], []
        for item in gui_state.camera_traj_list:
            target_c2ws.append(item["w2c"])
            assert item["img_wh"] == input_wh
            K = np.array(item["K"]).reshape(3, 3) / np.array([W, H, 1])[:, None]
            target_Ks.append(K)

        target_c2ws = torch.as_tensor(
                    np.linalg.inv(np.array(target_c2ws).reshape(-1, 4, 4)))
        target_Ks = torch.as_tensor(np.array(target_Ks).reshape(-1, 3, 3))
        return target_c2ws, target_Ks

    def get_target_c2ws_and_Ks_from_preset(
        self,
        preprocessed: dict,
        preset_traj: basic_trajectories,
        num_frames: int,
        zoom_factor: float | None,
    ):
        img_wh = preprocessed["input_wh"]
        start_c2w = preprocessed["input_c2ws"][0]
        start_w2c = torch.linalg.inv(start_c2w)
        look_at = torch.tensor([0, 0, 10])
        start_fov = DEFAULT_FOV_RAD
        target_c2ws, \
        target_fovs = get_preset_pose_fov(preset_traj, num_frames, start_w2c, look_at,
                                            -start_c2w[:3, 1], start_fov,
                                            spiral_radii=[1.0, 1.0, 0.5],
                                            zoom_factor=zoom_factor)
        target_c2ws = torch.as_tensor(target_c2ws)
        target_fovs = torch.as_tensor(target_fovs)
        target_Ks = get_default_intrinsics(target_fovs,
                                           aspect_ratio=img_wh[0]/img_wh[1])
        return target_c2ws, target_Ks

    def export_output_data(self, preprocessed: dict, output_dir: str):
        input_imgs = preprocessed["input_imgs"]
        input_wh   = preprocessed["input_wh"]
        input_Ks   = preprocessed["input_Ks"]
        input_c2ws = preprocessed["input_c2ws"]

        target_c2ws, target_Ks = self.get_target_c2ws_and_Ks_from_gui(preprocessed)

        num_inputs = len(input_imgs)
        num_targets = len(target_c2ws)

        input_imgs = (input_imgs.cpu().numpy() * 255.0).astype(np.uint8)
        input_c2ws = input_c2ws.cpu().numpy()
        input_Ks   =   input_Ks.cpu().numpy()
        target_c2ws = target_c2ws.cpu().numpy()
        target_Ks   =   target_Ks.cpu().numpy()
        img_whs = np.array(input_wh)[None].repeat(len(input_imgs) + len(target_Ks), 0)

        if os.path.isdir(output_dir) is False:
            os.makedirs(output_dir)

        img_paths = []
        for i, img in enumerate(input_imgs):
            iio.imwrite(img_path := osp.join(output_dir, f"{i:03d}.png"), img)
            img_paths.append(img_path)
        for i in range(num_targets):
            iio.imwrite(
                img_path := osp.join(output_dir, f"{i + num_inputs:03d}.png"),
                np.zeros((input_wh[1], input_wh[0], 3), dtype=np.uint8),
            )
            img_paths.append(img_path)

        # Convert from OpenCV to OpenGL camera format.
        all_c2ws = np.concatenate([input_c2ws, target_c2ws])
        all_Ks   = np.concatenate([input_Ks  , target_Ks  ])
        all_c2ws = all_c2ws @ np.diag([1, -1, -1, 1])

        create_transforms_simple(output_dir, img_paths, img_whs, all_c2ws, all_Ks)
        split_dict = {
            "train_ids": list(range(num_inputs)),
            "test_ids": list(range(num_inputs, num_inputs + num_targets)),
        }

        with open(osp.join(output_dir, f"train_test_split_{num_inputs}.json"), "w") as f:
            json.dump(split_dict, f, indent=4)

    def render(
        self,
        preprocessed: dict,
        seed: int,
        chunk_strategy: str,
        cfg: float,
        preset_traj: basic_trajectories | None,
        num_frames: int | None,
        zoom_factor: float | None,
        camera_scale: float,
        session_hash: str = None,
        run_gradio: bool = False,
    ):
        render_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        render_dir = osp.join(WORK_DIR, render_name)

        input_imgs = preprocessed["input_imgs"]
        input_c2ws = preprocessed["input_c2ws"]
        input_Ks = preprocessed["input_Ks"]
        input_wh = preprocessed["input_wh"]

        (W, H) = input_wh
        num_inputs = len(input_imgs)

        if preset_traj is None:
            target_c2ws, target_Ks = self.get_target_c2ws_and_Ks_from_gui(preprocessed)
        else:
            assert num_frames is not None
            assert num_inputs == 1
            input_c2ws = torch.eye(4)[None].to(dtype=input_c2ws.dtype)
            target_c2ws, \
            target_Ks = self.get_target_c2ws_and_Ks_from_preset(preprocessed, preset_traj, num_frames, zoom_factor)

        all_c2ws = torch.cat([input_c2ws, target_c2ws], 0)
        all_Ks   = torch.cat([input_Ks  , target_Ks  ], 0) * input_Ks.new_tensor([W, H, 1])[:, None]

        num_targets = len(target_c2ws)
        input_indices = list(range(num_inputs))
        target_indices = np.arange(num_inputs, num_inputs + num_targets).tolist()

        # Get anchor cameras.
        T = VERSION_DICT["T"]
        version_dict = copy.deepcopy(VERSION_DICT)
        num_anchors = infer_prior_stats(T, num_inputs,
                                        num_total_frames=num_targets,
                                        version_dict=version_dict)

        # infer_prior_stats modifies T in-place.
        T = version_dict["T"]
        assert isinstance(num_anchors, int)
        anchor_indices = np.linspace(num_inputs, 
                                     num_inputs + num_targets - 1,
                                     num_anchors).tolist()
        anchor_c2ws = all_c2ws[[round(ind) for ind in anchor_indices]]
        anchor_Ks   = all_Ks  [[round(ind) for ind in anchor_indices]]

        # Create image conditioning.
        all_imgs_np = F.pad(input_imgs, (0, 0, 0, 0, 0, 0, 0, num_targets), value=0.0).numpy()
        all_imgs_np = (all_imgs_np * 255.0).astype(np.uint8)
        image_cond = {
                      "img": all_imgs_np,
            "input_indices": input_indices,
            "prior_indices": anchor_indices,
        }

        # Create camera conditioning (K is unnormalized).
        camera_cond = {
            "c2w": all_c2ws,
              "K": all_Ks,
            "input_indices": list(range(num_inputs + num_targets)),
        }

        # Run rendering.
        num_steps = 50
        options_ori = VERSION_DICT["options"]
        options = copy.deepcopy(options_ori)
        options["chunk_strategy"] = chunk_strategy
        options["video_save_fps"] = 30.0
        options["beta_linear_start"] = 5e-6
        options["log_snr_shift"] = 2.4
        options["guider_types"] = [1, 2]
        options["cfg"] = [float(cfg), 3.0 if num_inputs >= 9 else 2.0]  
        # We define semi-dense-view regime to have 9 input views.

        options["camera_scale"] = camera_scale
        options["num_steps"] = num_steps
        options["cfg_min"] = 1.2
        options["encoding_t"] = 1
        options["decoding_t"] = 1

        # assert session_hash in ABORT_EVENTS
        # abort_event = ABORT_EVENTS[session_hash]
        # abort_event.clear()
        # options["abort_event"] = abort_event

        task = "img2trajvid"

        # Get number of first pass chunks.
        T_first_pass = T[0] if isinstance(T, (list, tuple)) else T
        chunk_strategy_first_pass = options.get("chunk_strategy_first_pass", "gt-nearest")

        num_chunks_0 = len(
            chunk_input_and_test(
                T_first_pass,
                input_c2ws,
                anchor_c2ws,
                input_indices,
                image_cond["prior_indices"],
                options={**options, "sampler_verbose": False},
                task=task,
                chunk_strategy=chunk_strategy_first_pass,
                gt_input_inds=list(range(input_c2ws.shape[0])),
            )[1]
        )

        # Get number of second pass chunks.
        anchor_argsort = np.argsort(input_indices + anchor_indices).tolist()
        anchor_indices =   np.array(input_indices + anchor_indices)[anchor_argsort].tolist()
        
        gt_input_inds = [anchor_argsort.index(i) for i in range(input_c2ws.shape[0])]
        anchor_c2ws_second_pass = torch.cat([input_c2ws, anchor_c2ws], dim=0)[anchor_argsort]

        T_second_pass = T[1] if isinstance(T, (list, tuple)) else T
        chunk_strategy = options.get("chunk_strategy", "nearest")
        num_chunks_1 = len(
            chunk_input_and_test(
                T_second_pass,
                anchor_c2ws_second_pass,
                target_c2ws,
                anchor_indices,
                target_indices,
                options={**options, "sampler_verbose": False},
                task=task,
                chunk_strategy=chunk_strategy,
                gt_input_inds=gt_input_inds,
            )[1]
        )

        if run_gradio:
            import gradio as gr
            first_pass_pbar = gr.Progress().tqdm(iterable=None, total=num_chunks_0*num_steps, desc="1st-pass sampling")
            second_pass_pbar = gr.Progress().tqdm(iterable=None, total=num_chunks_1*num_steps, desc="2nd-pass sampling")
        else:
            from tqdm import tqdm
            first_pass_pbar = tqdm(iterable=None, total=num_chunks_0*num_steps, desc="1st-pass sampling")
            second_pass_pbar = tqdm(iterable=None, total=num_chunks_1*num_steps, desc="2nd-pass sampling")

        video_path_generator = run_one_scene(
            task=task,
            version_dict={
                "H": H,
                "W": W,
                "T": T,
                "C": VERSION_DICT["C"],
                "f": VERSION_DICT["f"],
                "options": options,
            },
            model=MODEL,
            ae=AE,
            conditioner=CONDITIONER,
            denoiser=DENOISER,
            image_cond=image_cond,
            camera_cond=camera_cond,
            save_path=render_dir,
            use_traj_prior=True,
            traj_prior_c2ws=anchor_c2ws,
            traj_prior_Ks=anchor_Ks,
            seed=seed,
            gradio=run_gradio,
            first_pass_pbar=first_pass_pbar,
            second_pass_pbar=second_pass_pbar,
            abort_event=abort_event,
        )

        if not run_gradio:
            return video_path_generator

        output_queue = queue.Queue()

        import gradio as gr
        from gradio.context import LocalContext

        blocks = LocalContext.blocks.get()
        event_id = LocalContext.event_id.get()

        def worker():
            # gradio doesn't support threading with progress intentionally, so
            # we need to hack this.
            LocalContext.blocks.set(blocks)
            LocalContext.event_id.set(event_id)
            for i, video_path in enumerate(video_path_generator):
                if i == 0:
                    output_queue.put((video_path, gr.update(), gr.update(), gr.update()))
                elif i == 1:
                    output_queue.put((video_path, gr.update(visible=True), 
                                                  gr.update(visible=False),
                                                  gr.update(visible=False)))
                else:
                    gr.Error("More than 2 passes during rendering.")

        import threading
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while thread.is_alive() or not output_queue.empty():
            if abort_event.is_set():
                thread.join()
                abort_event.clear()
                yield (
                    gr.update(),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
            time.sleep(0.1)
            while not output_queue.empty():
                yield output_queue.get()

