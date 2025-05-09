import torch

from .txt2pano import Text2PanoPipeline
from ... import shared


if __name__ == "__main__":

    model_id = "sd15/diffusion360"
    model_path = shared.MVDIFF_LOCAL_MODELS[model_id]

    refine = True
    upscale = True

    positive_prompt = 'photorealistic, trend on artstation, ((best quality)), ((high res))'
    negative_prompt = 'worst quality, low quality, logo, text, watermark, monochrome, blur, '\
                      '(pillar), (human), complex texture, small objects, complex lighting'

    prompt  = "((job fair entrance)) leading to great hall with small brand booths"\
              ", futuristic style, dark space theme, dark galaxy theme, open space"\
              ", wooden floor, wood-liked striped ceiling with shining lightbulb"
    prompt += ", featuring planets and skyscrapers, a blue carpet leads into the archway"
    # prompt += ", archway is illuminated with vibrant nebulae and planets, overlooking a futuristic city skyline"
    # prompt += ", futuristic cityscape backdrop, cosmic elements (planets, nebula), blue and purple color palette, sleek archway design, blue carpet, art deco inspired"

    # Preprocess
    positive_prompt = prompt + ', ' + positive_prompt
    output_name = 'diffusion360_txt2pano'

    # Load pipeline
    pipe = Text2PanoPipeline(model_path, refine=refine, upscale=upscale)
    generator = torch.manual_seed(1995)
    
    # Inference 1: Generate panorama
    output_x1 = pipe.generate(positive_prompt, negative_prompt, height=768, width=1536)
    output_x1.save(f'./temp/{output_name}_x1.png')

    if pipe.pipe_sr is None:
        quit()
    
    # Inference 2: Refine with upscale x2
    image = output_x1.resize((1536 * 2, 768 * 2))
    output_x2 = pipe.refine(positive_prompt, negative_prompt, image)
    output_x2.save(f'./temp/{output_name}_x2.png')

    if pipe.upsampler is None:
        quit()

    # Inference 3: Refine with upscale x4
    image = output_x2
    image = image.resize((1536 * 2, 768 * 2))
    image = pipe.upsample(image)
    image = image.resize((1536 * 4, 768 * 4))

    output_x4 = pipe.refine(positive_prompt, negative_prompt, image)
    output_x4.save(f'./temp/{output_name}_x4.png')



