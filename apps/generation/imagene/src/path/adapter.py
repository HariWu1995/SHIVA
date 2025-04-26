# Full: 
#   https://huggingface.co/lllyasviel/sd_control_collection/tree/main

IMAGENE_ADAPTERS_FULL = {

    # T2I-Adapter
    "sdxl/line/sketch"   : "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_sketch.safetensors",
    "sdxl/line/canny"    : "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_canny.safetensors",
    "sdxl/line/lineart"  : "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_lineart.safetensors",
    "sdxl/pose/openpose" : "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_openpose.safetensors",
    "sdxl/depth/midas"   : "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_depth_midas.safetensors",
    "sdxl/depth/zoe"     : "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_depth_zoe.safetensors",

}

# FP-16: 
#   https://huggingface.co/webui/ControlNet-modules-safetensors/tree/main

IMAGENE_ADAPTERS_HALF = {

    # T2I-Adapter
    "sd15/line/canny"   : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_canny-fp16.safetensors",
    "sd15/line/sketch"  : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_sketch-fp16.safetensors",
    "sd15/pose/openpose": "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_openpose-fp16.safetensors",
    "sd15/pose/keypose" : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_keypose-fp16.safetensors",
    "sd15/depth/depth"  : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_depth-fp16.safetensors",
    "sd15/color/color"  : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_color-fp16.safetensors",
    "sd15/style/style"  : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_style-fp16.safetensors",
    "sd15/segment/seg"  : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_seg-fp16.safetensors",

}


# IP-Adapter
#   Repo: https://github.com/tencent-ailab/IP-Adapter
#   Checkpoint: https://huggingface.co/h94/IP-Adapter
#               https://huggingface.co/h94/IP-Adapter-FaceID

IP_ADAPTERS = {

    "sd15/ip/ip_adapter"       : "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors",
    "sd15/ip/ip_adapter_vitG"  : "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-G.safetensors",
    "sd15/ip/ip_adapter_light" : "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light.safetensors",
    "sd15/ip/ip_adapter_plus"  : "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors",
    "sd15/ip/ip_adapter_+face" : "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors",
    "sd15/ip/ip_adapter_fface" : "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors",

    "sdxl/ip/ip_adapter"       : "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors",
    "sdxl/ip/ip_adapter_vitH"  : "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
    "sdxl/ip/ip_adapter_plus"  : "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
    "sdxl/ip/ip_adapter_+face" : "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",

    # https://huggingface.co/InstantX/CSGO
    "sdxl/ip/csgo_c4_s16"   : "https://huggingface.co/InstantX/CSGO/resolve/main/csgo.bin",
    "sdxl/ip/csgo_c4_s32"   : "https://huggingface.co/InstantX/CSGO/resolve/main/csgo_4_32.bin",
}

