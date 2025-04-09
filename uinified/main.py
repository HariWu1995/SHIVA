import gradio as gr
import importlib
import os

APPS_DIR = "apps"

with gr.Blocks(title="Unified AI UI") as demo:
    for app_name in os.listdir(APPS_DIR):
        app_path = f"{APPS_DIR}.{app_name}.app_ui"
        try:
            module = importlib.import_module(app_path)
            ui_func = getattr(module, "get_ui")
            with gr.Tab(app_name.replace('_', ' ').title()):
                ui_func()  # Call to add block layout
        except (ImportError, AttributeError) as e:
            print(f"Skipping UI for {app_name}: {e}")

if __name__ == "__main__":
    demo.launch()
