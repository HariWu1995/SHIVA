from typing import List, Literal, Optional, Tuple, Union
from PIL import Image

import os
import math
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

from .io import decode_output, save_output
from .utils import randomize_seed, seed_everything
from .samplers import create_samplers, do_sample
from .sampling import DDPMDiscretization, DiscreteDenoiser
from .modules import AutoEncoder, CLIPConditioner
from .model import SGMWrapper, load_model
from .eval import (
    IS_TORCH_NIGHTLY, 
    load_img_and_K, 
    transform_img_and_K,
    chunk_input_and_test, assemble,
    replace_or_include_input_for_dict,
    get_value_dict, extend_dict, 
)

from ..path import SDIFF_LOCAL_MODELS, CAMCTRL_LOCAL_MODELS


if IS_TORCH_NIGHTLY:
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    COMPILE = True
else:
    COMPILE = False

VERSION_DICT = {
    "H": 576,
    "W": 576,
    "T": 21,
    "C": 4,
    "f": 8,
    "options": {},
}


def load_pipeline(
    base_model_name: str = "sd21_base",
    camera_model_name: str = "sd21/seva",
):
    sd_model_path  =   SDIFF_LOCAL_MODELS[base_model_name]
    cam_model_path = CAMCTRL_LOCAL_MODELS[camera_model_name]

    model = load_model(cam_model_path, device="cpu", verbose=True)
    model_wrapper = SGMWrapper(model)

    auto_encoder = AutoEncoder(sd_model_path, chunk_size=1)

    conditioner = CLIPConditioner().to(device=shared.device)
    model_wrapper = model_wrapper.to(device=shared.device)
    auto_encoder = auto_encoder.to(device=shared.device)

    discretizer = DDPMDiscretization()
    ddenoiser = DiscreteDenoiser(discretization=discretizer, num_idx=1000, device=shared.device)

    if COMPILE:
        conditioner = torch.compile(conditioner, dynamic=False)
        auto_encoder = torch.compile(auto_encoder, dynamic=False)
        model_wrapper = torch.compile(model_wrapper, dynamic=False)

    return model_wrapper, auto_encoder, conditioner, ddenoiser


def inference(
    # Task
    task,

    # Modules
    model,
    vaencoder,
    conditioner,
    denoiser,

    # Conditions
    image_cond,
    camera_cond,
    use_prior_traj,
    traj_prior_Ks,
    traj_prior_c2ws,
    seed: int = -1,

    # Misc.
    version_dict=VERSION_DICT,
    save_path=None,
    gradio=False,
    abort_event=None,
    first_pass_pbar=None,
    second_pass_pbar=None,
):
    H = version_dict["H"]
    W = version_dict["W"]
    T = version_dict["T"]
    C = version_dict["C"]
    F = version_dict["f"]
    opt = version_dict["options"]

    if isinstance(image_cond, str):
        image_cond = {"img": [image_cond]}

    imgs = []
    imgs_clip = []
    img_size = None

    for i, (img, K) in enumerate(zip(image_cond["img"], camera_cond["K"])):

        if isinstance(img, str) or img is None:
            img, K = load_img_and_K(img or img_size, None, K=K, device="cpu")  # type: ignore
            img_size = img.shape[-2:]
            
            if opt.get("L_short", -1) == -1:
                img, K = transform_img_and_K(
                    img,
                    (W, H),
                    K=K[None],
                    mode=(opt.get("transform_input", "crop") if i in image_cond["input_indices"]
                     else opt.get("transform_target", "crop")),
                    scale=(opt.get("transform_scale", 1.0) if i not in image_cond["input_indices"] else 1.0),
                )

            else:
                downsample = 3
                assert opt["L_short"] % (F * 2**downsample) == 0, \
                    f"Short side of image MUST be divisible by {F * 2**downsample}."

                img, K = transform_img_and_K(
                    img,
                    opt["L_short"],
                    K=K[None],
                    size_stride=F * 2**downsample,
                    mode=(opt.get("transform_input", "crop") if i in image_cond["input_indices"]
                     else opt.get("transform_target", "crop")),
                    scale=(opt.get("transform_scale", 1.0) if i not in image_cond["input_indices"] else 1.0),
                )

                version_dict["W"] = W = img.shape[-1]
                version_dict["H"] = H = img.shape[-2]

            K = K[0]
            K[0] /= W
            K[1] /= H

            img_clip = img
            camera_cond["K"][i] = K

        elif isinstance(img, np.ndarray):
            img_size = torch.Size(img.shape[:2])
            img = torch.as_tensor(img).permute(2, 0, 1)
            img = img.unsqueeze(0)
            img = img / 255.0 * 2.0 - 1.0

            if not gradio:
                img, K = transform_img_and_K(img, (W, H), K=K[None])
                assert K is not None
                K = K[0]

            K[0] /= W
            K[1] /= H
            camera_cond["K"][i] = K
            img_clip = img
        
        else:
            assert False, f"Variable `img` got {type(img)} type which is not supported!!!"
        
        imgs_clip.append(img_clip)
        imgs.append(img)

    imgs_clip = torch.cat(imgs_clip, dim=0)
    imgs = torch.cat(imgs, dim=0)

    if traj_prior_Ks is not None:
        assert img_size is not None
        for i, prior_k in enumerate(traj_prior_Ks):
            img, prior_k = load_img_and_K(img_size, None, K=prior_k, device="cpu")  # type: ignore
            img, prior_k = transform_img_and_K(
                img, 
                (W, H),
                K=prior_k[None],
                mode=opt.get("transform_target", "crop"),  # mode for prior is always same as target
                scale=opt.get("transform_scale", 1.0),  # scale for prior is always same as target
            )

            prior_k = prior_k[0]
            prior_k[0] /= W
            prior_k[1] /= H
            traj_prior_Ks[i] = prior_k

    opt["num_frames"] = T
    discretizer = denoiser.discretization
    torch.cuda.empty_cache()

    if seed <= 0:
        seed = randomize_seed()
    seed_everything(seed)

    # Get Data
    input_indices = image_cond["input_indices"]
    input_imgs      = imgs     [input_indices]
    input_imgs_clip = imgs_clip[input_indices]
    input_c2ws = camera_cond["c2w"][input_indices]
    input_Ks   = camera_cond[ "K" ][input_indices]

    test_indices = [i for i in range(len(imgs)) if i not in input_indices]
    test_imgs      = imgs     [test_indices]
    test_imgs_clip = imgs_clip[test_indices]
    test_c2ws = camera_cond["c2w"][test_indices]
    test_Ks   = camera_cond[ "K" ][test_indices]

    if opt.get("save_input", True):
        save_output(
            {"/image": input_imgs},
            save_path=os.path.join(save_path, "input"),
            video_save_fps=2,
        )

    if not use_traj_prior:
        chunk_strategy = opt.get("chunk_strategy", "gt")
        (
            _,
            input_inds_per_chunk, input_sels_per_chunk,
             test_inds_per_chunk,  test_sels_per_chunk,
        ) = chunk_input_and_test(
            T,
            input_c2ws,
            test_c2ws,
            input_indices,
            test_indices,
            options=opt,
            task=task,
            chunk_strategy=chunk_strategy,
            gt_input_inds=list(range(input_c2ws.shape[0])),
        )

        print(f"One pass - chunking with `{chunk_strategy}` strategy: total {len(input_inds_per_chunk)} forward(s) ...")

        all_samples = {}
        all_test_inds = []

        device = input_imgs.device
        pbar = tqdm(enumerate(zip(input_inds_per_chunk, input_sels_per_chunk,
                                   test_inds_per_chunk, test_sels_per_chunk)),
                        total=len(input_inds_per_chunk), leave=False)

        for i, (chunk_input_inds, chunk_input_sels,
                chunk_test_inds, chunk_test_sels) in pbar:
            (
                curr_input_sels, curr_test_sels,
                curr_input_maps, curr_test_maps,
            ) = pad_indices(
                chunk_input_sels,
                chunk_test_sels,
                T=T,
                padding_mode=opt.get("t_padding_mode", "last"),
            )

            curr_imgs, curr_imgs_clip, curr_c2ws, curr_Ks = [
                assemble(
                    input=x[chunk_input_inds],
                     test=y[chunk_test_inds],
                    input_maps=curr_input_maps,
                     test_maps=curr_test_maps,
                )
                for x, y in zip(
                    [
                        torch.cat([input_imgs     , get_k_from_dict(all_samples, "samples-rgb").to(device)], dim=0),
                        torch.cat([input_imgs_clip, get_k_from_dict(all_samples, "samples-rgb").to(device)], dim=0),
                        torch.cat([input_c2ws, test_c2ws[all_test_inds]], dim=0),
                        torch.cat([input_Ks  , test_Ks  [all_test_inds]], dim=0),
                    ],  # procedually append generated prior views to the input views
                    [
                        test_imgs, 
                        test_imgs_clip, 
                        test_c2ws, 
                        test_Ks],
                )
            ]

            image_sels_pad  = [s for (i, s) in zip(np.array(chunk_test_inds)[curr_test_maps[curr_test_maps != -1]], curr_test_sels) if test_indices[i] in  image_cond["input_indices"]]
            camera_sels_pad = [s for (i, s) in zip(np.array(chunk_test_inds)[curr_test_maps[curr_test_maps != -1]], curr_test_sels) if test_indices[i] in camera_cond["input_indices"]]

            value_dict = get_value_dict(
                curr_imgs.to("cuda"),
                curr_imgs_clip.to("cuda"),
                curr_input_sels + image_sels_pad,
                curr_c2ws,
                curr_Ks,
                curr_input_sels + camera_sels_pad,
                all_c2ws=camera_cond["c2w"],
                camera_scale=opt.get("camera_scale", 2.0),
            )

            samplers = create_samplers(opt["guider_types"], discretizer, [len(curr_imgs)],
                                       opt["num_steps"], 
                                       opt["cfg_min"], abort_event=abort_event)
            assert len(samplers) == 1
            samples = do_sample(model, vaencoder, conditioner, denoiser, samplers[0], value_dict,
                                H, W, C, F, T=len(curr_imgs), cfg=(opt["cfg"][0] if isinstance(opt["cfg"], (list, tuple))
                                                              else opt["cfg"]),
                                **{k: opt[k] for k in opt if k not in ["cfg", "T"]})
            samples = decode_output(samples, len(curr_imgs), chunk_test_sels)  # decode into dict
    
            if opt.get("save_first_pass", False):
                save_output(
                    replace_or_include_input_for_dict(samples, chunk_test_sels, curr_imgs, curr_c2ws, curr_Ks),
                    save_path=os.path.join(save_path, "first-pass", f"forward_{i}"),
                    video_save_fps=2,
                )
            extend_dict(all_samples, samples)
            all_test_inds.extend(chunk_test_inds)
    else:
        assert traj_prior_c2ws is not None, (
            "`traj_prior_c2ws` MUST be set when using 2-pass sampling. "
            "One potential reason is that the amount of input frames is larger than T. "
            "Set `num_prior_frames` manually to overwrite the infered stats."
        )

        traj_prior_c2ws = torch.as_tensor(traj_prior_c2ws, device=input_c2ws.device, dtype=input_c2ws.dtype)
        if traj_prior_Ks is None:
            traj_prior_Ks = test_Ks[:1].repeat_interleave(traj_prior_c2ws.shape[0], dim=0)

        traj_prior_imgs      = imgs     .new_zeros(traj_prior_c2ws.shape[0], *imgs     .shape[1:])
        traj_prior_imgs_clip = imgs_clip.new_zeros(traj_prior_c2ws.shape[0], *imgs_clip.shape[1:])

        #################################################
        #                   first pass                  #
        #################################################
        T_1st_pass = T[0] if isinstance(T, (list, tuple)) else T
        T_2nd_pass = T[1] if isinstance(T, (list, tuple)) else T

        chunk_strategy_1pass = opt.get("chunk_strategy_first_pass", "gt-nearest")
        (
            _,
            input_inds_per_chunk, input_sels_per_chunk,
            prior_inds_per_chunk, prior_sels_per_chunk,
        ) = chunk_input_and_test(
            T_1st_pass,
            input_c2ws,
            traj_prior_c2ws,
            input_indices,
            image_cond["prior_indices"],
            options=opt,
            task=task,
            chunk_strategy=chunk_strategy_1pass,
            gt_input_inds=list(range(input_c2ws.shape[0])),
        )
    
        print(f"1 / 2 passes - chunking strategy: `{chunk_strategy_1pass}` - total: {len(input_inds_per_chunk)} ...")

        all_samples = {}
        all_prior_inds = []

        pbar = tqdm(enumerate(zip(input_inds_per_chunk, input_sels_per_chunk,
                                  prior_inds_per_chunk, prior_sels_per_chunk)),
                        total=len(input_inds_per_chunk), leave=False)

        for i, (chunk_input_inds, chunk_input_sels,
                chunk_prior_inds, chunk_prior_sels) in pbar:
            (
                curr_input_sels, curr_prior_sels,
                curr_input_maps, curr_prior_maps,
            ) = pad_indices(
                chunk_input_sels,
                chunk_prior_sels,
                T=T_1st_pass,
                padding_mode=opt.get("t_padding_mode", "last"),
            )

            curr_imgs, curr_imgs_clip, curr_c2ws, curr_Ks = [
                assemble(
                    input=x[chunk_input_inds],
                    test=y[chunk_prior_inds],
                    input_maps=curr_input_maps,
                    test_maps=curr_prior_maps,
                )
                for x, y in zip(
                    [
                        torch.cat([input_imgs     , get_k_from_dict(all_samples, "samples-rgb").to(device)], dim=0),
                        torch.cat([input_imgs_clip, get_k_from_dict(all_samples, "samples-rgb").to(device)], dim=0),
                        torch.cat([input_c2ws, traj_prior_c2ws[all_prior_inds]], dim=0),
                        torch.cat([input_Ks  , traj_prior_Ks  [all_prior_inds]], dim=0),
                    ],  # procedually append generated prior views to the input views
                    [
                        traj_prior_imgs,
                        traj_prior_imgs_clip,
                        traj_prior_c2ws,
                        traj_prior_Ks,
                    ],
                )
            ]

            value_dict = get_value_dict(
                curr_imgs.to("cuda"),
                curr_imgs_clip.to("cuda"),
                curr_input_sels,
                curr_c2ws,
                curr_Ks,
                list(range(T_1st_pass)),
                all_c2ws=camera_cond["c2w"],
                camera_scale=opt.get("camera_scale", 2.0),
            )

            samplers = create_samplers(opt["guider_types"], discretizer, [T_1st_pass, T_2nd_pass],
                                       opt["num_steps"], 
                                       opt["cfg_min"], abort_event=abort_event)
            sampler_id = (
                1 if len(samplers) > 1 
                    and opt.get("ltr_first_pass", False)
                    and chunk_strategy_1pass != "gt"
                    and i > 0
                else 0
            )
    
            samples = do_sample(model, vaencoder, conditioner, denoiser, samplers[sampler_id],
                                value_dict, H, W, C, F, 
                                cfg=(opt["cfg"][0] if isinstance(opt["cfg"], (list, tuple))
                                else opt["cfg"]),
                                T=T_1st_pass,
                                global_pbar=first_pass_pbar,
                                **{k: opt[k] for k in opt if k not in ["cfg", "T", "sampler"]})
            if samples is None:
                return

            samples = decode_output(samples, T_1st_pass, chunk_prior_sels)  # decode into dict
            extend_dict(all_samples, samples)
            all_prior_inds.extend(chunk_prior_inds)

        if opt.get("save_first_pass", True):
            save_output(
                all_samples,
                save_path=os.path.join(save_path, "first-pass"),
                video_save_fps=5,
            )
            video_path_0 = os.path.join(save_path, "first-pass", "samples-rgb.mp4")
            yield video_path_0

        #################################################
        #                   second pass                 #
        #################################################
        prior_indices = image_cond["prior_indices"]
        assert (prior_indices is not None), \
            "`prior_frame_indices` needs to be set if using 2-pass sampling."

        prior_argsort = np.argsort(input_indices + prior_indices).tolist()
        prior_indices = np.array  (input_indices + prior_indices)[prior_argsort].tolist()
        gt_input_inds = [prior_argsort.index(i) for i in range(input_c2ws.shape[0])]

        traj_prior_imgs      = torch.cat([input_imgs     , get_k_from_dict(all_samples, "samples-rgb")], dim=0)[prior_argsort]
        traj_prior_imgs_clip = torch.cat([input_imgs_clip, get_k_from_dict(all_samples, "samples-rgb")], dim=0)[prior_argsort]

        traj_prior_c2ws = torch.cat([input_c2ws, traj_prior_c2ws], dim=0)[prior_argsort]
        traj_prior_Ks   = torch.cat([input_Ks  , traj_prior_Ks  ], dim=0)[prior_argsort]

        update_kv_for_dict(all_samples, "samples-rgb" , traj_prior_imgs)
        update_kv_for_dict(all_samples, "samples-c2ws", traj_prior_c2ws)
        update_kv_for_dict(all_samples, "samples-intrinsics", traj_prior_Ks)

        chunk_strategy = opt.get("chunk_strategy", "nearest")
        (
            _,
            prior_inds_per_chunk, prior_sels_per_chunk,
             test_inds_per_chunk,  test_sels_per_chunk,
        ) = chunk_input_and_test(
            T_2nd_pass,
            traj_prior_c2ws,
            test_c2ws,
            prior_indices,
            test_indices,
            options=opt,
            task=task,
            chunk_strategy=chunk_strategy,
            gt_input_inds=gt_input_inds,
        )

        print(f"2 / 2 passes - chunking strategy: `{chunk_strategy}` - total: {len(prior_inds_per_chunk)} ...")

        all_samples = {}
        all_test_inds = []

        pbar = tqdm(enumerate(zip(prior_inds_per_chunk, prior_sels_per_chunk,
                                   test_inds_per_chunk,  test_sels_per_chunk)),
                        total=len(prior_inds_per_chunk), leave=False)

        for i, (chunk_prior_inds, chunk_prior_sels,
                 chunk_test_inds,  chunk_test_sels) in pbar:
            (
                curr_prior_sels, curr_test_sels,
                curr_prior_maps, curr_test_maps,
            ) = pad_indices(
                chunk_prior_sels,
                chunk_test_sels,
                T=T_2nd_pass,
                padding_mode="last",
            )

            curr_imgs, curr_imgs_clip, curr_c2ws, curr_Ks = [
                assemble(
                    input=x[chunk_prior_inds],
                    test=y[chunk_test_inds],
                    input_maps=curr_prior_maps,
                    test_maps=curr_test_maps,
                )
                for x, y in zip(
                    [traj_prior_imgs, traj_prior_imgs_clip, traj_prior_c2ws, traj_prior_Ks],
                    [test_imgs, test_imgs_clip, test_c2ws, test_Ks],
                )
            ]

            value_dict = get_value_dict(
                curr_imgs.to("cuda"),
                curr_imgs_clip.to("cuda"),
                curr_prior_sels,
                curr_c2ws,
                curr_Ks,
                list(range(T_2nd_pass)),
                all_c2ws=camera_cond["c2w"],
                camera_scale=opt.get("camera_scale", 2.0),
            )

            samples = do_sample(model, vaencoder, conditioner, denoiser,
                                samplers[1] if len(samplers) > 1 else samplers[0],
                                value_dict, H, W, C, F,
                                T=T_2nd_pass,
                                cfg=(opt["cfg"][1] if isinstance(opt["cfg"], (list, tuple)) and len(opt["cfg"]) > 1
                                else opt["cfg"]),
                                global_pbar=second_pass_pbar,
                                **{k: opt[k] for k in opt if k not in ["cfg", "T", "sampler"]})
            if samples is None:
                return

            samples = decode_output(samples, T_2nd_pass, chunk_test_sels)  # decode into dict
            if opt.get("save_second_pass", False):
                save_output(
                    replace_or_include_input_for_dict(samples, chunk_test_sels, curr_imgs, curr_c2ws, curr_Ks),
                    save_path=os.path.join(save_path, "second-pass", f"forward_{i}"),
                    video_save_fps=2,
                )

            extend_dict(all_samples, samples)
            all_test_inds.extend(chunk_test_inds)

        all_samples = {
                key: value[np.argsort(all_test_inds)] 
            for key, value in all_samples.items()
        }

    if opt.get("replace_or_include_input", False):
        samples = replace_or_include_input_for_dict(all_samples, 
                                                    test_indices, imgs.clone(),
                                                    camera_cond["c2w"].clone(),
                                                    camera_cond["K"].clone())    
    else: 
        samples = all_samples

    save_output(
        samples,
        save_path=save_path,
        video_save_fps=opt.get("video_save_fps", 2),
    )

    video_path_1 = os.path.join(save_path, "samples-rgb.mp4")
    yield video_path_1


