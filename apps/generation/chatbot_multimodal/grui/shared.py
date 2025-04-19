"""
Reference:
    https://github.com/oobabooga/text-generation-webui
    https://github.com/oobabooga/text-generation-webui/blob/main/extensions/multimodal/README.md
"""
from copy import deepcopy


# Server variables
auto_launch = True
# share = False
# listen = False
# host = "localhost"
# port = 8192


# UI variables
gradio = {}
persistent_interface_state = {}
need_restart = False
multi_user = False

input_elements = []
reload_inputs = []
default_inputs = []

extensions = [
    "multimodal",
    "long_replies",
    # "google_translate",
    # "captioning",
    # "character_bias",
    # "character_gallery",
]


def add_extension(name, last=False):
    if extensions is None:
        extensions = [name]
    elif last:
        extensions = [x for x in extensions if x != name]
        extensions.append(name)
    elif name not in extensions:
        extensions.append(name)


# UI defaults
settings = {
    'character': 'SHIVA',
    'show_controls': True,
    'start_with': '',
    'mode': 'chat-instruct',
    'chat_style': 'cai-chat',
    'chat-instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',
    'custom_system_message': '',
    'prompt-default': 'QA',
    'prompt-notebook': 'QA',
    'name1': 'You',
    'user_bio': '',
    'preset': 'min_p',
    'max_new_tokens': 512,
    'max_new_tokens_min': 1,
    'max_new_tokens_max': 4096,
    'prompt_lookup_num_tokens': 0,
    'max_tokens_second': 0,
    'max_updates_second': 0,
    'auto_max_new_tokens': True,
    'skip_special_tokens': True,
    'ban_eos_token': False,
    'add_bos_token': True,
    'stream': True,
    'static_cache': False,
    'truncation_length': 8192,
    'seed': -1,
    'custom_stopping_strings': '',
    'custom_token_bans': '',
    'show_after': '',
    'negative_prompt': '',
    'autoload_model': False,
    'dark_theme': True,
    'default_extensions': [],
    'instruction_template_str': "{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not ns.found -%}\n    {{- '' + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + '\\n\\n' -}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- '' + message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{-'### Instruction:\\n' + message['content'] + '\\n\\n'-}}\n        {%- else -%}\n            {{-'### Response:\\n' + message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{-'### Response:\\n'-}}\n{%- endif -%}",
    'chat_template_str': "{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {%- if message['content'] -%}\n            {{- message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n        {%- if user_bio -%}\n            {{- user_bio + '\\n\\n' -}}\n        {%- endif -%}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{- name1 + ': ' + message['content'] + '\\n'-}}\n        {%- else -%}\n            {{- name2 + ': ' + message['content'] + '\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}",
}

default_settings = deepcopy(settings)


# (Long) Description
element_description = dict(
    chat_mode = 'Defines how the chat prompt is generated. In instruct and chat-instruct modes, the instruction template Parameters > Instruction template is used.',
    chat_instruct_cmd = '<|character|> and <|prompt|> get replaced with the bot name and the regular chat prompt respectively.',
    instruction_template = 'This gets autodetected; you usually don\'t need to change it. Used in instruct and chat-instruct modes.',
    custom_system_message = 'If not empty, will be used instead of the default one.',
    model_autoload = 'Whether to load the model as soon as it is selected in the Model dropdown.',
    model_download = "Enter the Hugging Face `username/model-name`, for instance: facebook/galactica-125m. To specify a branch, add it at the end after a \":\" character like this: facebook/galactica-125m:main. To download a single file, enter its name in the second box.",
    model_customization = "This allows you to set a customized template for the model currently selected in the \"Model loader\" menu. Whenever the model gets loaded, this template will be used in place of the template specified in the model's medatada, which sometimes is wrong.",
    llamacpp_hf_creation = "This will move your gguf file into a subfolder of `models` along with the necessary tokenizer files.",
    extension_menu = 'Note that some of these extensions may require manually installing Python requirements through the command: pip install -r extensions/extension_name/requirements.txt',
    extension_install = 'Enter the GitHub URL below and press Enter. For a list of extensions, see: https://github.com/oobabooga/text-generation-webui-extensions ⚠️  WARNING ⚠️ : extensions can execute arbitrary code. Make sure to inspect their source code before activating them.',
)


#########################################
#           CSS and JavaScript          #
#########################################
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


CSS_FILES = [
    "main.css", 
    "NotoSans/stylesheet.css", 
    "katex/katex.min.css", 
    "highlightjs/highlightjs-copy.min.css",
]

css = ""
for fp in CSS_FILES:
    with open(ROOT_DIR / 'css' / fp, 'r') as f:
        css += f.read()


JS_FILES = [
    "main.js",
    "global_scope.js",
    "save_files.js",
    "switch_tabs.js",
    "show_controls.js",
    "update_big_picture.js",
    "dark_theme.js",
]

js = dict()
for fp in JS_FILES:
    fn = os.path.splitext()[0]
    with open(ROOT_DIR / 'js' / fp, 'r') as f:
        js[fn] = f.read()

audio_noti_filepath = str(ROOT_DIR / "assets/notification.mp3")
if os.path.isfile(audio_noti_filepath):
    js["audio_noti"] = "document.querySelector('#audio_notification audio')?.play();"
else:
    js["audio_noti"] = ""


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

# legacy Gradio colors, before the 12/2024 update.
legacy_theme = gr.themes.Default(
    font=['Noto Sans', 'Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
).set(
    border_color_primary='#c5c5d2',
    button_large_padding='6px 12px',
    body_text_color_subdued='#484848',
    background_fill_secondary='#eaeaea',
    background_fill_primary='var(--neutral-50)',
    body_background_fill="white",
    block_background_fill="#f4f4f4",
    body_text_color="#333",
    button_secondary_background_fill="#f4f4f4",
    button_secondary_border_color="var(--border-color-primary)"
)

