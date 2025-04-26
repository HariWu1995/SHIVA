
IMAGENE_REMOTE_MODELS = {

    # Flux Family (~24Gb, except Flux-mini ~6.4Gb after distillation)
    "flux/mini" : "https://huggingface.co/TencentARC/flux-mini",
    "flux/dev"  : "https://huggingface.co/black-forest-labs/FLUX.1-dev",
    "flux/fill" : "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev",
    "flux/redux": "https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev",

    # BrushNet (2.4 / 1.4 Gb but require SD backbone)
    "brush/random_mask"          : "https://drive.google.com/drive/folders/1hCYIjeRGx3Zk9WZtQf0s3nDGfeiwqTsN",
    "brush/random_mask_xl"       : "https://drive.google.com/drive/folders/1H2annwRr1HkUppbHe2gt9HHqO59EHXKc",
    "brush/segmentation_mask"    : "https://drive.google.com/drive/folders/1KPFFYblnovk4MU74OCBfS1EZU_jhBsse",
    "brush/segmentation_mask_xl" : "https://drive.google.com/drive/folders/1twv3gFka6RQ27tqHwVw0ocrv7qKk_q5r",

    # Stable Diffusion 1.5 (2-4Gb)
    "sd15/sd_15_pruned_ema_only"    : "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
    "sd15/sd_15_pruned_ema_inpaint" : "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt",

    "sd15/dreamshaper_v8_inpaint"  : "https://civitai.com/api/download/models/134084?type=Model&format=SafeTensor&size=full&fp=fp32",
    "sd15/dreamshaper_v8"          : "https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16",

    "sd15/abs_reality_v18_inpaint" : "https://civitai.com/api/download/models/131004?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "sd15/abs_reality_v18"         : "https://civitai.com/api/download/models/132760?type=Model&format=SafeTensor&size=pruned&fp=fp16",

    "sd15/real_vision_v51_inpaint" : "https://civitai.com/api/download/models/501286?type=Model&format=SafeTensor&size=full&fp=fp16",
    "sd15/real_vision_v51"         : "https://civitai.com/api/download/models/501240?type=Model&format=SafeTensor&size=full&fp=fp16",

    "sd15/epic_photogasm"          : "https://civitai.com/api/download/models/429454?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "sd15/epic_photogasm_inpaint"  : "https://civitai.com/api/download/models/201346?type=Model&format=SafeTensor&size=pruned&fp=fp16",

    "sd15/icbinp"          : "https://civitai.com/api/download/models/667760?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "sd15/icbinp_inpaint"  : "https://civitai.com/api/download/models/306943?type=Model&format=SafeTensor&size=pruned&fp=fp16",

    "sd15/anylora"          : "https://civitai.com/api/download/models/95489?type=Model&format=SafeTensor&size=full&fp=fp16",
    "sd15/anylora_inpaint"  : "https://civitai.com/api/download/models/131256?type=Model&format=SafeTensor&size=full&fp=fp16",

    "sd15/never_ending_dream"         : "https://civitai.com/api/download/models/64094?type=Model&format=SafeTensor&size=full&fp=fp16",
    "sd15/never_ending_dream_inpaint" : "https://civitai.com/api/download/models/74750?type=Model&format=SafeTensor&size=full&fp=fp16",

    "sd15/disney_pixar_cartoon" : "https://civitai.com/api/download/models/69832?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "sd15/xxmix_9realistic"     : "https://civitai.com/api/download/models/102222?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "sd15/perfect_deliberate"   : "https://civitai.com/api/download/models/253055?type=Model&format=SafeTensor&size=full&fp=fp32",

    # Stable Diffusion XL (6-7Gb)
    "sdxl/sdxl_base_v1"   : "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
    "sdxl/sdxl_refine_v1" : "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors",
    
    "sdxl/sdxl_inpaint_v1"          : "https://huggingface.co/alexgenovese/checkpoint/resolve/021e192bd744c48a85f8ae1832662e77beb9aac7/sdxl-inpainting-1.0.safetensors",
    "sdxl/real_stock_photo_v2"      : "https://huggingface.co/alexgenovese/checkpoint/resolve/021e192bd744c48a85f8ae1832662e77beb9aac7/realisticStockPhoto_v20.safetensors",
    "sdxl/product_photo_midjourney" : "https://huggingface.co/alexgenovese/checkpoint/resolve/021e192bd744c48a85f8ae1832662e77beb9aac7/product-photography-midjourney.safetensors",

    "sdxl/icbinp" : "https://civitai.com/api/download/models/399481?type=Model&format=SafeTensor&size=pruned&fp=fp16",

    "sdxl/dreamshaper_light_inpaint" : "https://civitai.com/api/download/models/450187?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "sdxl/dreamshaper_light"         : "https://civitai.com/api/download/models/354657?type=Model&format=SafeTensor&size=full&fp=fp16",
    
    "sdxl/epic_realism_v16" : "https://civitai.com/api/download/models/1522905?type=Model&format=SafeTensor&size=pruned&fp=fp16",

    "sdxl/ahavietnam_realistic_v2" : "https://civitai.com/api/download/models/137827?type=Model&format=SafeTensor&size=full&fp=fp16",
    "sdxl/sdvn_real_detail_face"   : "https://civitai.com/api/download/models/134461?type=Model&format=SafeTensor&size=full&fp=fp16",

}


