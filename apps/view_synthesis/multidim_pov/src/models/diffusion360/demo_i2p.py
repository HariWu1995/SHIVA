from PIL import Image, ImageDraw

import torch
from diffusers.utils import load_image

from .img2pano import Image2PanoPipeline
from ..utils import resize_and_center_crop
from ... import shared


def load_and_resize_image(image_path, size: int = 768):
    image = load_image(image_path)
    image = image.resize((768, 768))
    image = resize_and_center_crop(image, 768)
    return image


if __name__ == "__main__":

    model_id = "sd15/diffusion360"
    model_path = shared.MVDIFF_LOCAL_MODELS[model_id]

    refine = True
    upscale = False

    positive_prompt = 'photorealistic, realistic textures, artstation, best quality, ultra res, 8k rendering, '\
                      'wide angle view, immersive scenery, immense, vast extent'
    negative_prompt = 'worst quality, low quality, logo, text, watermark, monochrome, blur, blurry, '\
                      'human, person, small object, (pillar), (obstacle), (barriers), restricted, '\
                      'closed space, complex texture, complex lighting, oversaturated, deformed, disfigured'

    # Example
    # NOTE: `mask` must be x2 width comparing to `image`
    # prompt = 'The office room'
    # image = load_image("./temp/i2p-image.jpg").resize((512, 512))
    # mask = load_image("./temp/i2p-mask.jpg")

    # Inputs
    # prompt = 'a modern conference stage with LED screen, overhead spotlights'
    # image_path = "C:/Users/Mr. RIAH/Pictures/_stage/stage-05.png"

    prompt  = "((a backdrop of a launching event)) at an immense exhibition hall, "\
              "open-ended long lobby on two sides, wooden elegant minimal interior, "\
              "symmetrical layout, wooden wall, wooden floor, high ceiling"

    # prompt  = "((a backdrop)) at the center of immense great hall, "\
    #           "futuristic exhibition booths along the long walls, "\
    # prompt += ", featuring planets and skyscrapers, a blue carpet leads into the archway"
    # prompt += ", archway is illuminated with vibrant nebulae and planets, overlooking a futuristic city skyline"
    # prompt += ", futuristic cityscape backdrop, cosmic elements (planets, nebula), blue and purple color palette, sleek archway design, blue carpet, art deco inspired"

    image_path = "C:/Users/Mr. RIAH/Pictures/_gate/welcome-gate-001-L.png"
    # image_path = "./temp/00003.png"

    image = load_and_resize_image(image_path, size=768)
    # image.save('./temp/image.png')

    # Preprocess
    positive_prompt = prompt + ', ' + positive_prompt
    output_name = 'diffusion360_img2pano'

    # Load pipeline
    pipe = Image2PanoPipeline(model_path, refine=refine, upscale=upscale)
    
    # Inference 1: Generate panorama 
    # 1024 x  512: ~ 0h 0m 21s
    # 2048 x 1024: ~ 0h 1m 16s
    # 4096 x 2048: > 3h 30 

    for seed in [303, 1995]:
        generator = torch.manual_seed(seed)
        output_x1 = pipe.generate(positive_prompt, negative_prompt, image, height=768, width=1536, generator=generator)
        output_x1.save(f'./temp/{output_name}_{seed:05d}_x1.png')

        if pipe.pipe_sr is None:
            continue

        # Inference 2: Refine with upscale x2 (~ 10m 43s)
        output_x1 = output_x1.resize((1536 * 2, 768 * 2))
        output_x2 = pipe.refine(positive_prompt, negative_prompt, output_x1)
        output_x2.save(f'./temp/{output_name}_{seed:05d}_x2.png')

        if pipe.upsampler is None:
            continue

        # Inference 3: Refine with upscale x4 (~ 1h 30m 43s)
        output_x2 = output_x2.resize((1536 * 2, 768 * 2))
        output_x2 = pipe.upsample(output_x2)
        output_x2 = output_x2.resize((1536 * 4, 768 * 4))

        output_x4 = pipe.refine(positive_prompt, negative_prompt, output_x2)
        output_x4.save(f'./temp/{output_name}_{seed:05d}_x4.png')

