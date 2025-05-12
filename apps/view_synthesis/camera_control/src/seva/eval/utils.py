import os
import torch


def is_k_in_dict(d, k):
    return any(map(lambda x: x.startswith(k), d.keys()))


def get_k_from_dict(d, k):
    media_d = {}
    for key, value in d.items():
        if key == k:
            return value
        if key.startswith(k):
            media = key.split("/")[-1]
            if media == "raw":
                return value
            media_d[media] = value
    if len(media_d) == 0:
        return torch.tensor([])
    assert (len(media_d) == 1), f"multiple media found in {d} for key {k}: {media_d.keys()}"
    return media_d[media]


def update_kv_for_dict(d, k, v):
    for key in d.keys():
        if key.startswith(k):
            d[key] = v
    return d


def extend_dict(ds, d):
    for key in d.keys():
        if key in ds:
            ds[key] = torch.cat([ds[key], d[key]], 0)
        else:
            ds[key] = d[key]
    return ds


def get_value_dict(
    curr_imgs,
    curr_imgs_clip,
    curr_input_frame_indices,
    curr_c2ws,
    curr_Ks,
    curr_input_camera_indices,
    all_c2ws,
    camera_scale,
):
    assert sorted(curr_input_camera_indices) == sorted(
        range(len(curr_input_camera_indices))
    )
    H, W, T, F = curr_imgs.shape[-2], curr_imgs.shape[-1], len(curr_imgs), 8

    value_dict = {}
    value_dict["cond_frames_without_noise"] = curr_imgs_clip[curr_input_frame_indices]
    value_dict["cond_frames"] = curr_imgs + 0.0 * torch.randn_like(curr_imgs)
    value_dict["cond_frames_mask"] = torch.zeros(T, dtype=torch.bool)
    value_dict["cond_frames_mask"][curr_input_frame_indices] = True
    value_dict["cond_aug"] = 0.0

    c2w = to_hom_pose(curr_c2ws.float())
    w2c = torch.linalg.inv(c2w)

    # camera centering
    ref_c2ws = all_c2ws

    camera_dist_2med = torch.norm(
        ref_c2ws[:, :3, 3] - ref_c2ws[:, :3, 3].median(0, keepdim=True).values,
        dim=-1,
    )

    valid_mask = camera_dist_2med <= torch.clamp(
        torch.quantile(camera_dist_2med, 0.97) * 10,
        max=1e6,
    )

    c2w[:, :3, 3] -= ref_c2ws[valid_mask, :3, 3].mean(0, keepdim=True)
    w2c = torch.linalg.inv(c2w)

    # camera normalization
    camera_dists = c2w[:, :3, 3].clone()

    translation_scaling_factor = (
               camera_scale
          if   torch.isclose(torch.norm(camera_dists[0]), torch.zeros(1), atol=1e-5).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )

    w2c[:, :3, 3] *= translation_scaling_factor
    c2w[:, :3, 3] *= translation_scaling_factor

    value_dict["plucker_coordinate"] = get_plucker_coordinates(
        extrinsics_src=w2c[0],
        extrinsics=w2c,
        intrinsics=curr_Ks.float().clone(),
        target_size=(H // F, W // F),
    )

    value_dict["c2w"] = c2w
    value_dict["K"] = curr_Ks
    value_dict["camera_mask"] = torch.zeros(T, dtype=torch.bool)
    value_dict["camera_mask"][curr_input_camera_indices] = True

    return value_dict

