# Full: 
#   https://huggingface.co/lllyasviel/ControlNet/tree/main/models
#   https://huggingface.co/lllyasviel/sd_control_collection/tree/main

IMAGENE_CTRLNETS_FULL = {
    
    # "sd15/line/canny"    : "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth",
    # "sd15/line/hed"      : "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_hed.pth",
    # "sd15/line/mlsd"     : "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_mlsd.pth",
    # "sd15/line/scribble" : "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth",

    # "sd15/depth/depth"   : "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth",
    # "sd15/depth/normal"  : "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_normal.pth",

    # "sd15/pose/openpose" : "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth",
    # "sd15/segment/seg"   : "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_seg.pth",

    "sd15/upscale/tile"     : "https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/resolve/main/diffusion_pytorch_model.bin",

    "sd15/color/brightness" : "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/ioclab_sd15_recolor.safetensors",
    "sdxl/deblur/deblur"    : "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur.safetensors",

    "sdxl/line/canny"       : "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors",
    "sdxl/line/canny_mini"  : "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small/resolve/main/diffusion_pytorch_model.fp16.safetensors",
    "sdxl/line/misto"       : "https://huggingface.co/TheMistoAI/MistoLine/resolve/main/diffusion_pytorch_model.fp16.safetensors",

    "sdxl/depth/depth"      : "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors",
    "sdxl/depth/depth_mini" : "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-small/resolve/main/diffusion_pytorch_model.fp16.safetensors",
    "sdxl/depth/depth_zoe"  : "https://huggingface.co/diffusers/controlnet-zoe-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
    
    # NOTE: Omni-Controlnet - 
    #   Repo: https://github.com/xinsir6/ControlNetPlus
    #   Ckpt: https://huggingface.co/xinsir/controlnet-union-sdxl-1.0
    "sdxl/omni/union_pro"    : "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
    "sdxl/omni/union_promax" : "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors",

}

# FP-16: 
#   https://huggingface.co/webui/ControlNet-modules-safetensors/tree/main

IMAGENE_CTRLNETS_HALF = {
    
    "sd15/line/canny"    : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_canny-fp16.safetensors",
    "sd15/line/hed"      : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_hed-fp16.safetensors",
    "sd15/line/mlsd"     : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_mlsd-fp16.safetensors",
    "sd15/line/scribble" : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_scribble-fp16.safetensors",

    "sd15/depth/depth"   : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_depth-fp16.safetensors",
    "sd15/depth/normal"  : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_normal-fp16.safetensors",
    "sd15/segment/seg"   : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_seg-fp16.safetensors",

    "sd15/pose/openpose"        : "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_openpose-fp16.safetensors",
    "sd15/pose/openpose_animal" : "https://huggingface.co/huchenlei/animal_openpose/resolve/main/control_sd15_animal_openpose_fp16.pth",

    "sd15/other/outfit_variant" : "https://civitai.com/api/download/models/469517?type=Model&format=SafeTensor",
    "sd15/other/qrcode_monster" : "https://civitai.com/api/download/models/122143?type=Model&format=SafeTensor",
    "sd15/other/qrcode_pattern" : "https://civitai.com/api/download/models/111973?type=Model&format=SafeTensor",
}

