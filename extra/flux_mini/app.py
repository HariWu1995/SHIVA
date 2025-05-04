import gradio as gr
import numpy as np

from .main import *


def wrap_inference(
    prompt, 
    seed=42, 
    randomize_seed=False, 
    width=1024, 
    height=1024, 
    guidance_scale=3.5, 
    num_inference_steps=28, 
    progress=gr.Progress(track_tqdm=True)
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    
    for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
        output_type="pil",
        good_vae=good_vae,
    ):
        yield img, seed

    
examples = [
    "thousands of luminous oysters on a shore reflecting and refracting the sunset",
    "profile of sad Socrates, full body, high detail, dramatic scene, Epic dynamic action, wide angle, cinematic, hyper realistic, concept art, warm muted tones as painted by Bernie Wrightson, Frank Frazetta,",
    "ghosts, astronauts, robots, cats, superhero costumes, line drawings, naive, simple, exploring a strange planet, coloured pencil crayons, , black canvas background, drawn by 5 year old child",
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# FLUX-Mini
        A 3.2B param rectified flow transformer distilled from [FLUX.1 [dev]](https://blackforestlabs.ai/)  
        [[non-commercial license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)] 
        """)
        
        with gr.Row():
            prompt = gr.Text(label="Prompt", show_label=False, max_lines=1, placeholder="Enter your prompt", container=False)
            run_button = gr.Button("Run", scale=0)
        
        result = gr.Image(label="Result", show_label=False)
        
        with gr.Accordion("Advanced Settings", open=False):
            
            seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=-1)
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            
            with gr.Row():
                width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=1024)
                height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=1024)
            
            with gr.Row():
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=15, step=0.1, value=3.5)
                inference_steps = gr.Slider(label="Inference steps", minimum=1, maximum=50, step=1, value=28)
        
        gr.Examples(
            examples = examples,
            fn = wrap_inference,
            inputs = [prompt],
            outputs = [result, seed],
            cache_examples="lazy"
        )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = wrap_inference,
        inputs = [prompt, seed, randomize_seed, width, height, guidance_scale, inference_steps],
        outputs = [result, seed]
    )

demo.launch()
