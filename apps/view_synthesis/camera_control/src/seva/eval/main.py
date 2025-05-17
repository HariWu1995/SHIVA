import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)

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


from .chunk import chunk_gt, chunk_nearest, chunk_interp

def chunk_input_and_test(
    T,
    input_c2ws,
    test_c2ws,
    input_ords = None, 
    test_ords = None,  
    task: str = "img2img",
    options: dict = {},
    chunk_strategy: str = "gt",
    gt_input_inds: list = [],
):
    M = input_c2ws.shape[0]
    N = test_c2ws.shape[0]

    # Supported: gt / gt-ltr / gt-nearest
    if chunk_strategy.startswith("gt"):
        assert len(gt_input_inds) < T, (
            f"Number of gt input frames {len(gt_input_inds)} should be "
            f"less than {T} when `gt` chunking strategy is used."
        )
        assert (list(range(M)) == gt_input_inds), \
            "All input_c2ws should be gt when `gt` chunking strategy is used."
        chunks = chunk_gt(chunk_strategy, gt_input_inds, N, T, options, test_c2ws)

    # Supported: nearest / nearest-gt / nearest-{number}
    elif chunk_strategy.startswith("nearest"):
        input_imgs = np.array([f"!{i:03d}" for i in range(M)])
        test_imgs  = np.array([f">{i:03d}" for i in range(N)])
        chunks = chunk_nearest(chunk_strategy, gt_input_inds, 
                                    input_imgs, test_imgs,
                                    input_c2ws, test_c2ws, N, T)

    # Supported: interp / interp-gt
    elif chunk_strategy.startswith("interp"):
        # `interp` chunk requires ordering info
        assert input_ords is not None \
            and test_ords is not None, \
            "For `interp` chunking strategy, order of input and test frames MUST be provided."
        chunks = chunk_interp(chunk_strategy, gt_input_inds, 
                             task, T, input_c2ws, test_c2ws, 
                                      input_ords, test_ords)
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
        input_inds_per_chunk,   # ordering of input in raw sequence
        input_sels_per_chunk,   # ordering of input in one-forward sequence of length T
        test_inds_per_chunk,    # ordering of test in raw sequence
        test_sels_per_chunk,    # oredering of test in one-forward sequence of length T
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
        json.dump(out, of, indent=4)


def assemble(inputs, tests, input_maps, test_maps):
    T = len(input_maps)
    assembled = torch.zeros_like(tests[-1:]).repeat_interleave(T, dim=0)
    assembled[input_maps != -1] = inputs[input_maps[input_maps != -1]]
    assembled[ test_maps != -1] =  tests[ test_maps[ test_maps != -1]]
    assert np.logical_xor(input_maps != -1, 
                           test_maps != -1).all()
    return assembled

