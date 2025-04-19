import yaml
import html
import gradio as gr
from deep_translator import GoogleTranslator


params = {
    "activate": True,
    "language": "vi",
}

# Reading data from the YAML file
with open('language_codes.yaml', 'r') as file:
    language_codes = yaml.safe_load(file)


def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """
    if not params['activate']:
        return string
    return GoogleTranslator(source=params['language'], target='en').translate(string)


def output_modifier(string):
    """
    This function is applied to the model outputs.
    """
    if not params['activate']:
        return string
    translated_str = GoogleTranslator(source='en', target=params['language']).translate(html.unescape(string))
    return html.escape(translated_str)


def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """
    return string


def create_ui():
    # Finding the language name from the language code to use as the default value
    language_name = list(language_codes.keys())[
                    list(language_codes.values()).index(params['language'])]

    # Gradio elements
    with gr.Row():
        activate = gr.Checkbox(value=params['activate'], label='Activate translation')

    with gr.Row():
        language = gr.Dropdown(value=language_name, 
                             choices=[k for k in language_codes], label='Language')

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    language.change(lambda x: params.update({"language": language_codes[x]}), language, None)

