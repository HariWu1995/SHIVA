import torch
import numpy as np


def replace_or_include_input_for_dict(samples, test_indices, imgs, c2w, K):
    samples_new = {}
    for sample, value in samples.items():

        if "rgb" in sample:
            imgs[test_indices] = (
           value[test_indices] if value.shape[0] == imgs.shape[0] else value).to(device=imgs.device, dtype=imgs.dtype)
            samples_new[sample] = imgs

        elif "c2w" in sample:
            c2w[test_indices] = (
          value[test_indices] if value.shape[0] == c2w.shape[0] else value).to(device=c2w.device, dtype=c2w.dtype)
            samples_new[sample] = c2w

        elif "intrinsics" in sample:
            K[test_indices] = (
        value[test_indices] if value.shape[0] == K.shape[0] 
                          else value).to(device = K.device, dtype = K.dtype)
            samples_new[sample] = K

        else:
            samples_new[sample] = value
    return samples_new


def decode_output(samples, T, indices=None):
    """
    decode model output into dict if it is not
    """
    # model with postprocessor and outputs dict
    if isinstance(samples, dict):
        for sample, value in samples.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            elif isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            else:
                value = torch.tensor(value)
            if indices is not None \
            and value.shape[0] == T:
                value = value[indices]
            samples[sample] = value

    # model without postprocessor and outputs tensor (rgb)
    else:
        samples = samples.detach().cpu()
        if indices is not None \
        and samples.shape[0] == T:
            samples = samples[indices]
        samples = {"samples-rgb/image": samples}
    return samples

