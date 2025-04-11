"""
Reference:
    https://www.gradio.app/docs/gradio/annotatedimage
    https://www.gradio.app/docs/gradio/imageeditor
"""
import os
from glob import glob
from PIL import Image, ImageDraw

import numpy as np
import gradio as gr

from ..src import all_models, default_model
from ..src.infer_auto import inference_auto as infer


def wrap_inference(img, model, points, pred_iou_thresh, box_nms_thresh, crop_nms_thresh):
    masks = infer(img, model, points, pred_iou_thresh, box_nms_thresh, crop_nms_thresh)
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    anns = [(m['segmentation'].astype(float), str(i)) for i, m in enumerate(masks)]
    return (img, anns)


# Define UI settings & layout

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)

    folder_examples = os.path.join(os.path.dirname(__file__), 'assets')
    image_examples = glob(os.path.join(folder_examples, "*.jpg"))
    video_examples = glob(os.path.join(folder_examples, "*.mp4"))
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## Auto Segmentation")
        
        with gr.Row():
            original_img = gr.Image(label="Original Image")
            annotated_img = gr.AnnotatedImage(label="Segmented Instances")
        
        with gr.Row():

            with gr.Column(scale=7):
                with gr.Accordion(label='Advanced Settings', open=False):
                    model = gr.Dropdown(all_models, value=default_model, label="Select Model")
                    points = gr.Number(value=32, label="points_per_side", precision=0,
                                        info='#points to be sampled along each side of the image.')
                    pred_iou_thresh = gr.Slider(value=0.88, minimum=0, maximum=1.0, step=0.01, label="pred_iou_thresh",
                                                info="A filtering threshold in [0,1], using the model's predicted mask quality.")
                    box_nms_thresh = gr.Slider(value=0.7, minimum=0, maximum=1.0, step=0.01, label="box_nms_thresh",
                                                info='The box IoU cutoff used by non-maximal ression to filter duplicate masks.')
                    crop_nms_thresh = gr.Slider(value=0.7, minimum=0, maximum=1.0, step=0.01, label="crop_nms_thresh",
                                                info='The box IoU cutoff used by NMS to filter duplicated masks between different crops.')
            with gr.Column(scale=1):
                button = gr.Button("Segment")

        # Events
        button.click(
                fn=wrap_inference, 
            inputs=[original_img, model, points, pred_iou_thresh, box_nms_thresh, crop_nms_thresh],
            outputs=[annotated_img]
        )
    
    return gui, annotated_img


if __name__ == "__main__":
    gui, _ = create_ui()
    gui.launch(server_name='localhost', server_port=8000)
