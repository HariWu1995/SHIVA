"""
Reference:
    https://www.gradio.app/docs/gradio/annotatedimage
    https://www.gradio.app/docs/gradio/imageeditor
    https://docs.ultralytics.com/models/sam/#sam-prediction-example
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py#L92
"""
import os
from glob import glob
from PIL import Image, ImageDraw

import numpy as np
import gradio as gr

from ..src import all_models, default_model
from ..src.infer_point import inference_point as infer


# points, color and marker
FG_POINTS = []     # Foreground Points to include
BG_POINTS = []     # Background Points to exclude
colors = [(255, 0, 0), (0, 255, 0)]
radius = 5


def draw(image):
    global FG_POINTS, BG_POINTS

    # Draw point on a copy of the image
    annotated = image.copy()
    drawer = ImageDraw.Draw(annotated)
    for px, py in FG_POINTS:
        drawer.ellipse((px - radius, py - radius, px + radius, py + radius), fill=colors[1])
    for px, py in BG_POINTS:
        drawer.ellipse((px - radius, py - radius, px + radius, py + radius), fill=colors[0])

    return annotated


def draw_points(image, excluded, event: gr.SelectData):
    global FG_POINTS, BG_POINTS
    if image is None:
        return None, None, "[]", "[]"

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Record the point
    x, y = event.index
    if excluded:
        BG_POINTS.append((x, y))
    else:
        FG_POINTS.append((x, y))

    annotated = draw(image)
    return annotated, str(FG_POINTS), str(BG_POINTS)


def undo_points(image, excluded):
    if image is None:
        return None, None, "[]", "[]"

    global FG_POINTS, BG_POINTS
    if excluded:
        BG_POINTS = BG_POINTS[:-1]
    else:
        FG_POINTS = FG_POINTS[:-1]

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    annotated = draw(image)

    return annotated, str(FG_POINTS), str(BG_POINTS)


def clear_points():
    global FG_POINTS, BG_POINTS
    FG_POINTS = []
    BG_POINTS = []
    return None, None, "[]", "[]"


def wrap_inference(img, model):
    global FG_POINTS, BG_POINTS
    points = FG_POINTS + BG_POINTS
    labels = [1] * len(FG_POINTS) + [0] * len(BG_POINTS)
    masks = infer(img, model, points, labels)
    anns = [(m.astype(float), str(i)) for i, m in enumerate(masks)]
    return (img, anns)


# Define UI settings & layout

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)

    folder_examples = os.path.join(os.path.dirname(__file__), 'assets')
    image_examples = glob(os.path.join(folder_examples, "*.jpg"))
    video_examples = glob(os.path.join(folder_examples, "*.mp4"))
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 📦 Instance Segmentation using Foreground + Background Points")
        
        with gr.Row():
            original_img = gr.Image(label="Original Image", interactive=True)
            pointed_img = gr.Image(label="Image with Points")
        
        with gr.Row():

            with gr.Column(scale=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        undo_btn = gr.Button("Undo", variant="secondary")
                        clear_btn = gr.Button("Clear", variant="secondary")
                        segm_btn = gr.Button("Segment", variant="primary")
                    with gr.Column(scale=8):
                        exclusion = gr.Checkbox(label="Background", info="Check to exclude as background")
                        fg_points = gr.Textbox(label="Foreground Points", interactive=False)
                        bg_points = gr.Textbox(label="background Points", interactive=False)
                with gr.Accordion(label='Advanced Settings', open=False):
                    model = gr.Dropdown(all_models, value=default_model, label="Select Model")

            with gr.Column(scale=1):
                annotated_img = gr.AnnotatedImage(label="Segmented Instances")

        # Events
        original_img.select(fn=draw_points, 
                        inputs=[original_img, exclusion], 
                        outputs=[pointed_img, fg_points, bg_points])

        undo_btn.click(fn=undo_points, 
                    inputs=[original_img, exclusion], 
                    outputs=[pointed_img, fg_points, bg_points])

        clear_btn.click(fn=clear_points, 
                    outputs=[original_img, pointed_img, fg_points, bg_points])

        segm_btn.click(
                fn=wrap_inference, 
            inputs=[original_img, model],
            outputs=[annotated_img]
        )
    
    return gui, annotated_img


if __name__ == "__main__":
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000)
