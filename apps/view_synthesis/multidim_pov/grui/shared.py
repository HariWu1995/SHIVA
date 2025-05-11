from pathlib import Path

# Server variables
allowed_paths = [Path(__file__).resolve().parents[4] / "_samples"]
need_restart = False
auto_launch = True
streaming = False
share = False
listen = False
host = "localhost"
port = 8192

# UI variables
gradio = {}
states = {}

symbols = dict(
    trigger = '🚀 Trigger',
    refiner = '⏯️ Refiner',
    refresh = '🔄 Refresh',
    reset = '🔄 Reset',
    release = '🗑️ Release',
    delete = '🗑️ Delete',
    offload = '⌛ Offload',
    save = '💾 Save',
    undo = '🔙 Undo',
)

wip_message = '*{user} is typing ...*'      # WIP: work in progress
generation_params = []

# UI defaults
settings = {
    'seed': -1,
    'stream': True,
    'static_cache': False,
    'negative_prompt': '',
    'autoload_model': False,
    'dark_theme': True,
}

from copy import deepcopy
default_settings = deepcopy(settings)

# (Long) Description
element_description = dict()


#############################
#           Theme           #
#############################
import gradio as gr

theme = gr.themes.Default(
    font=['Noto Sans', 'Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
).set(
    # General Colors
    border_color_primary='#c5c5d2',
    body_text_color_subdued='#484848',
    background_fill_secondary='#eaeaea',
    background_fill_secondary_dark='var(--selected-item-color-dark)',
    background_fill_primary='var(--neutral-50)',
    background_fill_primary_dark='var(--darker-gray)',
    body_background_fill="white",
    block_background_fill="transparent",
    body_text_color="#333",
    button_secondary_background_fill="#f4f4f4",
    button_secondary_border_color="var(--border-color-primary)",

    # Dark Mode Colors
    input_background_fill_dark='var(--darker-gray)',
    checkbox_background_color_dark='var(--darker-gray)',
    block_background_fill_dark='transparent',
    block_border_color_dark='transparent',
    input_border_color_dark='var(--border-color-dark)',
    checkbox_border_color_dark='var(--border-color-dark)',
    border_color_primary_dark='var(--border-color-dark)',
    button_secondary_border_color_dark='var(--border-color-dark)',
    body_background_fill_dark='var(--dark-gray)',
    button_primary_background_fill_dark='transparent',
    button_secondary_background_fill_dark='transparent',
    checkbox_label_background_fill_dark='transparent',
    button_cancel_background_fill_dark='transparent',
    button_secondary_background_fill_hover_dark='var(--selected-item-color-dark)',
    checkbox_label_background_fill_hover_dark='var(--selected-item-color-dark)',
    table_even_background_fill_dark='var(--darker-gray)',
    table_odd_background_fill_dark='var(--selected-item-color-dark)',
    code_background_fill_dark='var(--darker-gray)',

    # Shadows and Radius
    checkbox_label_shadow='none',
    block_shadow='none',
    block_shadow_dark='none',
    button_large_radius='0.375rem',
    button_large_padding='6px 12px',
    input_radius='0.375rem',
)
