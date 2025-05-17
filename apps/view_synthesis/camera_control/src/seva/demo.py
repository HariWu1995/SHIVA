import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:512"

from glob import glob
from tqdm import tqdm
from PIL import Image
from json import dumps

import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from .. import shared
from .eval import create_transforms_simple, infer_prior_stats
from .tasks import parse_task
from .config import VERSION_DICT
from .pipeline import inference, load_pipeline


def main(
    models: list | tuple,
    task: str,
    data_path: str,
    data_items: str | list | None = None,
    H=None, W=None, T=None,
    save_path: str = None,
    save_subdir: str = "",
    use_traj_prior: bool = False,
    **overwrite_options,
):
    if H is not None:
        VERSION_DICT["H"] = H
    if W is not None:
        VERSION_DICT["W"] = W
    if T is not None:
        VERSION_DICT["T"] = [int(t) for t in T.split(",")] if isinstance(T, str) else T

    options = VERSION_DICT["options"]
    options.update(overwrite_options)
    print("="*44, "\n", dumps(options, indent=4), "\n", "="*44)

    num_inputs = options["num_inputs"]
    seed = options["seed"]

    if save_path is None:
        save_path = data_path

    if data_items is not None:
        if not isinstance(data_items, (list, tuple)):
            data_items = data_items.split(",")
        scenes = [osp.join(data_path, item) for item in data_items]
    else:
        scenes = [s for s in glob(osp.join(data_path, "*")) if s.endswith(('.png','.jpg','.jpeg'))]

    for scene in tqdm(scenes):
        print(f'\n\nProcessing {scene} ...')

        scene_name = osp.splitext(osp.basename(scene))[0]
        save_path_scene = osp.join(save_path, task, save_subdir, scene_name)

        if options.get("skip_saved", True) \
        and osp.exists(
            osp.join(save_path_scene, "transforms.json")):
            print(f"Skipping {scene} as it is already sampled.")
            continue

        # parse_task 
        # -> infer_prior_stats modifies VERSION_DICT["T"] in-place.
        all_imgs_path, num_inputs, num_targets, \
                input_indices, anchor_indices, \
                c2ws, Ks, anchor_c2ws, anchor_Ks = parse_task(task, scene, num_inputs,
                                                              VERSION_DICT["T"], 
                                                              VERSION_DICT)
        assert num_inputs is not None

        # Create image conditioning.
        image_cond = {
                      "img": all_imgs_path,
            "input_indices": input_indices,
            "prior_indices": anchor_indices,
        }
    
        # Create camera conditioning.
        camera_cond = {
                      "c2w": c2ws.clone(),
                        "K": Ks.clone(),
            "input_indices": list(range(num_inputs + num_targets)),
        }

        try:
            # run_one_scene 
            # -> transform_img_and_K modifies VERSION_DICT["H"] and VERSION_DICT["W"] in-place.
            video_path_generator = inference(
                # Task & models
                task, *models,

                # Conditions
                image_cond=image_cond,
                camera_cond=camera_cond,
                use_prior_traj=use_traj_prior,
                traj_prior_c2ws=anchor_c2ws,
                traj_prior_Ks=anchor_Ks,

                # Misc.
                version_dict=VERSION_DICT,  # H, W maybe updated in-place
                save_path=save_path_scene,
                seed=seed,
            )

        except Exception:
            continue

        for _ in video_path_generator:
            pass

        # Convert from OpenCV to OpenGL camera format.
        c2ws = c2ws @ torch.tensor(np.diag([1, -1, -1, 1])).float()

        sample_paths = sorted(glob(osp.join(save_path_scene, "samples-rgb", "*.png")))
        input_paths  = sorted(glob(osp.join(save_path_scene, "input", "*.png")))

        if len(sample_paths) != len(c2ws):

            assert len(sample_paths) == num_targets
            assert len(input_paths) == num_inputs
            assert c2ws.shape[0] == num_inputs + num_targets

            target_indices = [i for i in range(c2ws.shape[0]) if i not in input_indices]
            input_paths = [
                      input_paths[ input_indices.index(i)] if i in input_indices
                else sample_paths[target_indices.index(i)] for i in range(c2ws.shape[0])
            ]

        create_transforms_simple(
            save_path=save_path_scene,
            img_paths=input_paths,
            img_whs=np.array([VERSION_DICT["W"], 
                              VERSION_DICT["H"]])[None].repeat(num_inputs + num_targets, 0),
            c2ws=c2ws,
            Ks=Ks,
        )


if __name__ == "__main__":

    # Load pipeline
    # model_wrapper, auto_encoder, conditioner, ddenoiser = load_pipeline()
    models = load_pipeline()

    # Test: img2img
    # main(
    #     models,
    #     task="img2img", 
    #     data_path="./temp/cam_pov",
    #     data_items=['glass_building','garden_flythrough'],
    #     num_steps=10,
    #     num_inputs=4, 
    #     video_save_fps=10,
    # )

    # Test: img2trajvid
    main(
        models,
        task="img2trajvid", 
        data_path="./temp/cam_pov",
        data_items=['custom_traj','garden_flythrough'],
        num_steps=10,
        num_inputs=4, 
    use_traj_prior=True,
    chunk_strategy="interp-gt",
    chunk_strategy_first_pass="gt-nearest",
        cfg=[3., 2.],
        video_save_fps=10,
    )

    # Test: img2trajvid_s-prob
    main(
        models,
        task="img2trajvid_s-prob", 
        data_path="./temp/cam_scene",
    chunk_strategy="interp",
        traj_prior="zoom-out",
    use_traj_prior=True,
        num_steps=25,
        num_targets=10,
        L_short=448,
        cfg=[4., 2.],
        guider=[1, 1],
        video_save_fps=10,
        replace_or_include_input=True,
    )

