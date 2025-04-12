from typing import List, Literal, Optional, Tuple, Union
from PIL import Image

import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

from .io import *
from .utils import *
from .samplers import *


def load_img_and_K(
    image_path_or_size: Union[str, torch.Size],
    size: Optional[Union[int, Tuple[int, int]]],
    scale: float = 1.0,
    center: Tuple[float, float] = (0.5, 0.5),
    K: torch.Tensor | None = None,
    size_stride: int = 1,
    center_crop: bool = False,
    image_as_tensor: bool = True,
    context_rgb: np.ndarray | None = None,
    device: str = "cuda",
):
    if isinstance(image_path_or_size, torch.Size):
        image = Image.new("RGBA", image_path_or_size[::-1])
    else:
        image = Image.open(image_path_or_size).convert("RGBA")

    w, h = image.size
    if size is None:
        size = (w, h)

    image = np.array(image).astype(np.float32) / 255
    if image.shape[-1] == 4:
        rgb   = image[:, :, :3]
        alpha = image[:, :, 3:]
        if context_rgb is not None:
            image = rgb * alpha + context_rgb * (1 - alpha)
        else:
            image = rgb * alpha + (1 - alpha)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).to(dtype=torch.float32)
    image = image.unsqueeze(0)

    if isinstance(size, (tuple, list)):
        # => if size is a tuple or list, we first rescale to fully cover the `size`
        # area and then crop the `size` area from the rescale image
        W, H = size
    else:
        # => if size is int, we rescale the image to fit the shortest side to size
        # => if size is None, no rescaling is applied
        W, H = get_wh_with_fixed_shortest_side(w, h, size)
    W = math.floor(W / size_stride + 0.5) * size_stride
    H = math.floor(H / size_stride + 0.5) * size_stride

    rfs = get_resizing_factor((math.floor(H * scale), math.floor(W * scale)), (h, w))
    resize_size = rh, rw = [int(np.ceil(rfs * s)) for s in (h, w)]
    image = F.interpolate(image, resize_size, mode="area", antialias=False)

    if scale < 1.0:
        pw = math.ceil((W - resize_size[1]) * 0.5)
        ph = math.ceil((H - resize_size[0]) * 0.5)
        image = F.pad(image, (pw, pw, ph, ph), "constant", 1.0)

    cy_center = int(center[1] * image.shape[-2])
    cx_center = int(center[0] * image.shape[-1])
    if center_crop:
        side = min(H, W)
        ct = max(0, cy_center - side // 2)
        cl = max(0, cx_center - side // 2)
        ct = min(ct, image.shape[-2] - side)
        cl = min(cl, image.shape[-1] - side)
        image = tvF.crop(image, top=ct, left=cl, height=side, width=side)
    else:
        ct = max(0, cy_center - H // 2)
        cl = max(0, cx_center - W // 2)
        ct = min(ct, image.shape[-2] - H)
        cl = min(cl, image.shape[-1] - W)
        image = tvF.crop(image, top=ct, left=cl, height=H, width=W)

    if K is not None:
        K = K.clone()
        if  torch.all(K[:2, -1] >= 0) \
        and torch.all(K[:2, -1] <= 1):
            K[:2] *= K.new_tensor([rw, rh])[:, None]  # normalized K
        else:
            K[:2] *= K.new_tensor([rw / w, rh / h])[:, None]  # unnormalized K
        K[:2, 2] -= K.new_tensor([cl, ct])

    if image_as_tensor:
        # tensor of shape (1, 3, H, W) with values ranging from (-1, 1)
        image = image.to(device) * 2.0 - 1.0
    else:
        # PIL Image with values ranging from (0, 255)
        image = image.permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).astype(np.uint8))
    return image, K


def transform_img_and_K(
    image: torch.Tensor,
    size: Union[int, Tuple[int, int]],
    scale: float = 1.0,
    center: Tuple[float, float] = (0.5, 0.5),
    K: torch.Tensor | None = None,
    size_stride: int = 1,
    mode: str = "crop",
):
    available_modes = ["crop","pad","stretch"]
    assert mode in available_modes, \
        f"mode should be one of {available_modes}, got {mode}"

    h, w = image.shape[-2:]
    if isinstance(size, (tuple, list)):
        # => if size is a tuple or list, we first rescale to fully cover the `size`
        # area and then crop the `size` area from the rescale image
        W, H = size
    else:
        # => if size is int, we rescale the image to fit the shortest side to size
        # => if size is None, no rescaling is applied
        W, H = get_wh_with_fixed_shortest_side(w, h, size)
    W = math.floor(W / size_stride + 0.5) * size_stride
    H = math.floor(H / size_stride + 0.5) * size_stride

    if mode == "stretch":
        rh, rw = H, W
    else:
        rfs = get_resizing_factor((H, W), (h, w), cover_target = (mode != "pad"))
        (rh, rw) = [int(np.ceil(rfs * s)) for s in (h, w)]

    rh, rw = int(rh / scale), int(rw / scale)
    image = F.interpolate(image, (rh, rw), mode="area", antialias=False)

    cy_center = int(center[1] * image.shape[-2])
    cx_center = int(center[0] * image.shape[-1])
    if mode != "pad":
        ct = max(0, cy_center - H // 2)
        cl = max(0, cx_center - W // 2)
        ct = min(ct, image.shape[-2] - H)
        cl = min(cl, image.shape[-1] - W)
        image = tvF.crop(image, top=ct, left=cl, height=H, width=W)
        pl, pt = 0, 0
    else:
        pt = max(0, H // 2 - cy_center)
        pl = max(0, W // 2 - cx_center)
        pb = max(0, H - pt - image.shape[-2])
        pr = max(0, W - pl - image.shape[-1])
        image = tvF.pad(image, [pl, pt, pr, pb])
        cl, ct = 0, 0

    if K is not None:
        K = K.clone()
        # K[:, :2, 2] += K.new_tensor([pl, pt])
        if  torch.all(K[:, :2, -1] >= 0) \
        and torch.all(K[:, :2, -1] <= 1):
            K[:, :2] *= K.new_tensor([rw, rh])[None, :, None]  # normalized K
        else:
            K[:, :2] *= K.new_tensor([rw / w, rh / h])[None, :, None]  # unnormalized K
        K[:, :2, 2] += K.new_tensor([pl - cl, pt - ct])

    return image, K


def run_one_scene(
    task,
    version_dict,
    model,
    ae,
    conditioner,
    denoiser,
    image_cond,
    camera_cond,
    save_path,
    use_traj_prior,
    traj_prior_Ks,
    traj_prior_c2ws,
    seed=23,
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
    options = version_dict["options"]

    if isinstance(image_cond, str):
        image_cond = {"img": [image_cond]}
    imgs_clip, imgs, img_size = [], [], None

    for i, (img, K) in enumerate(zip(image_cond["img"], camera_cond["K"])):

        if isinstance(img, str) or img is None:
            img, K = load_img_and_K(img or img_size, None, K=K, device="cpu")  # type: ignore
            img_size = img.shape[-2:]
            
            if options.get("L_short", -1) == -1:
                img, K = transform_img_and_K(
                    img,
                    (W, H),
                    K=K[None],
                    mode=(options.get("transform_input", "crop") if i in image_cond["input_indices"]
                     else options.get("transform_target", "crop")),
                    scale=(options.get("transform_scale", 1.0) if i not in image_cond["input_indices"] else 1.0),
                )

            else:
                downsample = 3
                assert options["L_short"] % F * 2**downsample == 0, (
                    f"Short side of the image should be divisible by F*2**{downsample}={F * 2**downsample}."
                )
                img, K = transform_img_and_K(
                    img,
                    options["L_short"],
                    K=K[None],
                    size_stride=F * 2**downsample,
                    mode=(options.get("transform_input", "crop") if i in image_cond["input_indices"]
                     else options.get("transform_target", "crop")),
                    scale=(options.get("transform_scale", 1.0) if i not in image_cond["input_indices"] else 1.0),
                )
                version_dict["W"] = W = img.shape[-1]
                version_dict["H"] = H = img.shape[-2]

            K = K[0]
            K[0] /= W
            K[1] /= H
            camera_cond["K"][i] = K
            img_clip = img

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
                mode=options.get("transform_target", "crop"),  # mode for prior is always same as target
                scale=options.get("transform_scale", 1.0),  # scale for prior is always same as target
            )
            prior_k = prior_k[0]
            prior_k[0] /= W
            prior_k[1] /= H
            traj_prior_Ks[i] = prior_k

    options["num_frames"] = T
    discretization = denoiser.discretization
    torch.cuda.empty_cache()

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

    if options.get("save_input", True):
        save_output(
            {"/image": input_imgs},
            save_path=os.path.join(save_path, "input"),
            video_save_fps=2,
        )

    if not use_traj_prior:
        chunk_strategy = options.get("chunk_strategy", "gt")
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
            options=options,
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
                padding_mode=options.get("t_padding_mode", "last"),
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
                camera_scale=options.get("camera_scale", 2.0),
            )

            samplers = create_samplers(options["guider_types"], discretization, [len(curr_imgs)],
                                       options["num_steps"], options["cfg_min"], abort_event=abort_event)
            assert len(samplers) == 1
            samples = do_sample(model, ae, conditioner, denoiser, samplers[0], value_dict,
                                H, W, C, F, T=len(curr_imgs), cfg=(options["cfg"][0] if isinstance(options["cfg"], (list, tuple))
                                                              else options["cfg"]),
                                **{k: options[k] for k in options if k not in ["cfg", "T"]})
            samples = decode_output(samples, len(curr_imgs), chunk_test_sels)  # decode into dict
    
            if options.get("save_first_pass", False):
                save_output(
                    replace_or_include_input_for_dict(samples, chunk_test_sels, curr_imgs, curr_c2ws, curr_Ks),
                    save_path=os.path.join(save_path, "first-pass", f"forward_{i}"),
                    video_save_fps=2,
                )
            extend_dict(all_samples, samples)
            all_test_inds.extend(chunk_test_inds)
    else:
        assert traj_prior_c2ws is not None, (
            "`traj_prior_c2ws` should be set when using 2-pass sampling. "
            "One potential reason is that the amount of input frames is larger than T. "
            "Set `num_prior_frames` manually to overwrite the infered stats."
        )
        traj_prior_c2ws = torch.as_tensor(traj_prior_c2ws, device=input_c2ws.device, dtype=input_c2ws.dtype)
        if traj_prior_Ks is None:
            traj_prior_Ks = test_Ks[:1].repeat_interleave(traj_prior_c2ws.shape[0], dim=0)

        traj_prior_imgs      = imgs     .new_zeros(traj_prior_c2ws.shape[0], *imgs     .shape[1:])
        traj_prior_imgs_clip = imgs_clip.new_zeros(traj_prior_c2ws.shape[0], *imgs_clip.shape[1:])

        # ---------------------------------- first pass ----------------------------------
        T_1st_pass  = T[0] if isinstance(T, (list, tuple)) else T
        T_2nd_pass = T[1] if isinstance(T, (list, tuple)) else T
        chunk_strategy_first_pass = options.get("chunk_strategy_first_pass", "gt-nearest")
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
            options=options,
            task=task,
            chunk_strategy=chunk_strategy_first_pass,
            gt_input_inds=list(range(input_c2ws.shape[0])),
        )
    
        print(
            f"Two passes (first) - chunking with `{chunk_strategy_first_pass}` strategy: total {len(input_inds_per_chunk)} forward(s) ..."
        )

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
                padding_mode=options.get("t_padding_mode", "last"),
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
                camera_scale=options.get("camera_scale", 2.0),
            )

            samplers = create_samplers(options["guider_types"], discretization, [T_1st_pass, T_2nd_pass],
                                       options["num_steps"], options["cfg_min"], abort_event=abort_event)
            sampler_id = (
                1 if len(samplers) > 1 
                    and options.get("ltr_first_pass", False)
                    and chunk_strategy_first_pass != "gt"
                    and i > 0
                else 0
            )
            samples = do_sample(model, ae, conditioner, denoiser, samplers[sampler_id],
                                value_dict, H, W, C, F, 
                                cfg=(options["cfg"][0] if isinstance(options["cfg"], (list, tuple))
                                else options["cfg"]),
                                T=T_1st_pass,
                                global_pbar=first_pass_pbar,
                                **{k: options[k] for k in options if k not in ["cfg", "T", "sampler"]})
            if samples is None:
                return
            samples = decode_output(samples, T_1st_pass, chunk_prior_sels)  # decode into dict
            extend_dict(all_samples, samples)
            all_prior_inds.extend(chunk_prior_inds)

        if options.get("save_first_pass", True):
            save_output(
                all_samples,
                save_path=os.path.join(save_path, "first-pass"),
                video_save_fps=5,
            )
            video_path_0 = os.path.join(save_path, "first-pass", "samples-rgb.mp4")
            yield video_path_0

        # ---------------------------------- second pass ----------------------------------
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

        chunk_strategy = options.get("chunk_strategy", "nearest")
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
            options=options,
            task=task,
            chunk_strategy=chunk_strategy,
            gt_input_inds=gt_input_inds,
        )
        print(
            f"Two passes (second) - chunking with `{chunk_strategy}` strategy: total {len(prior_inds_per_chunk)} forward(s) ..."
        )

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
                camera_scale=options.get("camera_scale", 2.0),
            )

            samples = do_sample(model, ae, conditioner, denoiser,
                                samplers[1] if len(samplers) > 1 else samplers[0],
                                value_dict, H, W, C, F,
                                T=T_2nd_pass,
                                cfg=(options["cfg"][1] if isinstance(options["cfg"], (list, tuple)) and len(options["cfg"]) > 1
                                else options["cfg"]),
                                global_pbar=second_pass_pbar,
                                **{k: options[k] for k in options if k not in ["cfg", "T", "sampler"]})
            if samples is None:
                return
            samples = decode_output(samples, T_2nd_pass, chunk_test_sels)  # decode into dict
            if options.get("save_second_pass", False):
                save_output(
                    replace_or_include_input_for_dict(samples, chunk_test_sels, curr_imgs, curr_c2ws, curr_Ks),
                    save_path=os.path.join(save_path, "second-pass", f"forward_{i}"),
                    video_save_fps=2,
                )
            extend_dict(all_samples, samples)
            all_test_inds.extend(chunk_test_inds)

        all_samples = {
            key: value[np.argsort(all_test_inds)] for key, value in all_samples.items()
        }

    if options.get("replace_or_include_input", False):
        samples = replace_or_include_input_for_dict(all_samples, 
                                                    test_indices, imgs.clone(),
                                                    camera_cond["c2w"].clone(),
                                                    camera_cond["K"].clone())    
    else: 
        samples = all_samples

    save_output(
        samples,
        save_path=save_path,
        video_save_fps=options.get("video_save_fps", 2),
    )

    video_path_1 = os.path.join(save_path, "samples-rgb.mp4")
    yield video_path_1


