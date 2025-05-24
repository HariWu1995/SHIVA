import torch


def merge_lora2unet(
    unet_dict,
    lora_dict,
    lora_scale: float = 1.0,
    lora_keys: list = ['to_q', 'to_k', 'to_v', 'to_out'],
    skip_keys: list = ['bias'],
):
    # Load base model
    # unet = UNet.from_pretrained(model_path, subfolder='unet')
    # fused_dict = unet.state_dict()

    # Load LoRA
    # lora_dict = torch.load(lora_path, map_location='cpu')
    # if 'state_dict' in lora_dict:
    #     lora_dict = lora_dict['state_dict']

    fused_dict = unet_dict

    # Merge LoRA
    used_lora_keys = []
    for lora_key in lora_keys:

        unet_keys = []
        for k in fused_dict.keys():
            if lora_key not in k:
                continue
            if any([sk in k for sk in skip_keys]):
                continue
            unet_keys.append(k)
        print(f'There are {len(unet_keys)} unet keys for lora key: {lora_key}')

        for unet_key in unet_keys:
            prefixes = unet_key.split('.')
            idx = prefixes.index(lora_key)

            lora_down_key = ".".join(prefixes[:idx]) + f".processor.{lora_key}_lora.down" + f".{prefixes[-1]}"
            lora_up_key   = ".".join(prefixes[:idx]) + f".processor.{lora_key}_lora.up"   + f".{prefixes[-1]}"

            assert lora_down_key in lora_dict, f"{lora_down_key} is expected in `lora_dict`"
            assert   lora_up_key in lora_dict,   f"{lora_up_key} is expected in `lora_dict`"

            print(f'Fusing lora weight for {unet_key}')
            fused_dict[unet_key] = \
            fused_dict[unet_key] + torch.bmm(lora_dict[ lora_up_key ][None, ...], 
                                                   lora_dict[lora_down_key][None, ...])[0] * lora_scale
            used_lora_keys.append(lora_down_key)
            used_lora_keys.append(lora_up_key)

    assert len(set(used_lora_keys) - set(lora_dict.keys())) == 0
    return fused_dict

