"""
Stable Virtual Camera
    https://github.com/Stability-AI/stable-virtual-camera/blob/main/demo_gr.py
"""
import gradio as gr
from gradio import networking
from gradio.context import LocalContext
from gradio.tunneling import CERTIFICATE_PATH, Tunnel

from ..src.viz_camera_traj import Renderer

