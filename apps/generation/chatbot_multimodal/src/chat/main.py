import re
import copy

import functools
from functools import partial
from datetime import datetime
from pathlib import Path
from PIL import Image

import html
import json
import base64

from .. import shared
from ..logging import logger
from ..generation import generate_reply, get_encoded_length, get_max_prompt_length
from ..utils import save_file, delete_file

from .utils import jinja_env, yaml, get_generation_prompt, replace_character_names


def generate_chat_prompt(user_input, state, **kwargs):
    impersonate = kwargs.get('impersonate', False)
    _continue = kwargs.get('_continue', False)
    return_rows = kwargs.get('return_rows', False)
    history = kwargs.get('history', state['history'])['internal']

    # Templates
    instr_template_str = state['instruction_template_str']
    chat_template_str = state['chat_template_str']

    if state['mode'] != 'instruct':
        chat_template_str = replace_character_names(chat_template_str, state['name1'], state['name2'])

    instr_template = jinja_env.from_string(instr_template_str)
    chat_template = jinja_env.from_string(chat_template_str)

    instruct_renderer = partial(
        instr_template.render,
        builtin_tools=None,
        tools=None,
        tools_in_user_message=False,
        add_generation_prompt=False
    )

    chat_renderer = partial(
        chat_template.render,
        add_generation_prompt=False,
        name1=state['name1'],
        name2=state['name2'],
        user_bio=replace_character_names(state['user_bio'], state['name1'], state['name2']),
    )

    messages = []

    if state['mode'] == 'instruct':
        renderer = instruct_renderer
        if state['custom_system_message'].strip() != '':
            messages.append({"role": "system", "content": state['custom_system_message']})
    else:
        renderer = chat_renderer
        if state['context'].strip() != '' or state['user_bio'].strip() != '':
            context = replace_character_names(state['context'], state['name1'], state['name2'])
            messages.append({"role": "system", "content": context})

    insert_pos = len(messages)
    for user_msg, assistant_msg in reversed(history):
        user_msg = user_msg.strip()
        assistant_msg = assistant_msg.strip()

        if assistant_msg:
            messages.insert(insert_pos, {"role": "assistant", "content": assistant_msg})

        if user_msg not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            messages.insert(insert_pos, {"role": "user", "content": user_msg})

    user_input = user_input.strip()
    if user_input and not impersonate and not _continue:
        messages.append({"role": "user", "content": user_input})

    def remove_extra_bos(prompt):
        for bos_token in ['<s>', '<|startoftext|>', '<BOS_TOKEN>', '<|endoftext|>']:
            while prompt.startswith(bos_token):
                prompt = prompt[len(bos_token):]

        return prompt

    def make_prompt(messages):
        if state['mode'] == 'chat-instruct' and _continue:
            prompt = renderer(messages=messages[:-1])
        else:
            prompt = renderer(messages=messages)

        if state['mode'] == 'chat-instruct':
            outer_messages = []
            if state['custom_system_message'].strip() != '':
                outer_messages.append({"role": "system", "content": state['custom_system_message']})

            prompt = remove_extra_bos(prompt)
            command = state['chat-instruct_command']
            command = command.replace('<|character|>', state['name2'] if not impersonate else state['name1'])
            command = command.replace('<|prompt|>', prompt)
            command = replace_character_names(command, state['name1'], state['name2'])

            if _continue:
                prefix = get_generation_prompt(renderer, impersonate=impersonate, strip_trailing_spaces=False)[0]
                prefix += messages[-1]["content"]
            else:
                prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]
                if not impersonate:
                    prefix = apply_extensions('bot_prefix', prefix, state)

            outer_messages.append({"role": "user", "content": command})
            outer_messages.append({"role": "assistant", "content": prefix})

            prompt = instr_template.render(messages=outer_messages)
            suffix = get_generation_prompt(instruct_renderer, impersonate=False)[1]
            if len(suffix) > 0:
                prompt = prompt[:-len(suffix)]

        else:
            if _continue:
                suffix = get_generation_prompt(renderer, impersonate=impersonate)[1]
                if len(suffix) > 0:
                    prompt = prompt[:-len(suffix)]
            else:
                prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]
                if state['mode'] == 'chat' and not impersonate:
                    prefix = apply_extensions('bot_prefix', prefix, state)

                prompt += prefix

        prompt = remove_extra_bos(prompt)
        return prompt

    prompt = make_prompt(messages)

    # Handle truncation
    if shared.tokenizer is not None:
        max_length = get_max_prompt_length(state)
        encoded_length = get_encoded_length(prompt)
        while len(messages) > 0 and encoded_length > max_length:

            # Remove old message, save system message
            if len(messages) > 2 and messages[0]['role'] == 'system':
                messages.pop(1)

            # Remove old message when no system message is present
            elif len(messages) > 1 and messages[0]['role'] != 'system':
                messages.pop(0)

            # Resort to truncating the user input
            else:

                user_message = messages[-1]['content']

                # Bisect the truncation point
                left, right = 0, len(user_message) - 1

                while right - left > 1:
                    mid = (left + right) // 2

                    messages[-1]['content'] = user_message[:mid]
                    prompt = make_prompt(messages)
                    encoded_length = get_encoded_length(prompt)

                    if encoded_length <= max_length:
                        left = mid
                    else:
                        right = mid

                messages[-1]['content'] = user_message[:left]
                prompt = make_prompt(messages)
                encoded_length = get_encoded_length(prompt)
                if encoded_length > max_length:
                    logger.error(f"Failed to build the chat prompt. The input is too long for the available context length.\n\nTruncation length: {state['truncation_length']}\nmax_new_tokens: {state['max_new_tokens']} (is it too high?)\nAvailable context length: {max_length}\n")
                    raise ValueError
                else:
                    logger.warning(f"The input has been truncated. Context length: {state['truncation_length']}, max_new_tokens: {state['max_new_tokens']}, available context length: {max_length}.")
                    break

            prompt = make_prompt(messages)
            encoded_length = get_encoded_length(prompt)

    if return_rows:
        return prompt, [message['content'] for message in messages]
    else:
        return prompt


def generate_chat_reply(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False):
    history = state['history']
    if regenerate or _continue:
        text = ''
        if (len(history['visible']) == 1 and not history['visible'][0][0]) or len(history['internal']) == 0:
            yield history
            return

    show_after = html.escape(state.get("show_after")) if state.get("show_after") else None
    for history in chatbot_wrapper(text, state, regenerate=regenerate, _continue=_continue, loading_message=loading_message, for_ui=for_ui):
        if show_after:
            after = history["visible"][-1][1].partition(show_after)[2] or "*Is thinking...*"
            yield {
                'internal': history['internal'],
                'visible': history['visible'][:-1] + [[history['visible'][-1][0], after]]
            }
        else:
            yield history


@functools.cache
def load_character_memoized(character, name1, name2):
    return load_character(character, name1, name2)


@functools.cache
def load_instruction_memoized(template):
    return load_instruction_template(template)
