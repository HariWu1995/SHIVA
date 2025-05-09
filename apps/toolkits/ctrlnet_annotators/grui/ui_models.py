import gradio as gr

from ..src import (
    all_options_segment,
    all_options_depth,
    all_options_edge,
    all_options_face,
    all_options_pose,
)


def load_annotators():
    all_annotators = ["basic_binary"]
    all_annotators.extend([f"edge_{x}" for x in all_options_edge])
    all_annotators.extend([f"face_{x}" for x in all_options_face])
    all_annotators.extend([f"pose_{x}" for x in all_options_pose])
    all_annotators.extend([f"depth_{x}" for x in all_options_depth])
    all_annotators.extend([f"segment_{x}" for x in all_options_segment])
    # all_annotators = sorted(all_annotators)

    return gr.Dropdown(label="Annotator", value="basic_binary", choices=all_annotators)

