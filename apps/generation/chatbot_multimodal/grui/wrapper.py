from copy import deepcopy
import gradio as gr
import html

from .html import convert_to_markdown

from ..src import shared
from ..src.instruct import template_dir, generate_instruction_template
from ..src.generation import generate_reply

from ..src.chat.main import generate_chat_prompt
from ..src.chat.utils import get_stopping_strings, character_is_loaded, redraw_html as redraw
from ..src.chat.message import remove_last_message, send_dummy_message, send_dummy_reply
from ..src.chat.history import find_all_histories_with_first_prompts, start_new_chat, load_latest_history
from ..src.chat.character import load_character, upload_profile_picture


#########################################################
#               Wrappers for Gradio-UI                  #
#########################################################

def chatbot_wrapper(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False):
    history = state['history']
    output = deepcopy(history)
    output = apply_extensions('history', output)
    state = apply_extensions('state', state)

    visible_text = None
    stopping_strings = get_stopping_strings(state)
    is_stream = state['stream']

    # Prepare the input
    if not (regenerate or _continue):
        visible_text = html.escape(text)

        # Apply extensions
        text, visible_text = apply_extensions('chat_input', text, visible_text, state)
        text = apply_extensions('input', text, state, is_chat=True)

        output['internal'].append([text, ''])
        output['visible'].append([visible_text, ''])

        # *Is typing...*
        if loading_message:
            yield {
                 'visible': output['visible'][:-1] + [[output['visible'][-1][0], shared.processing_message]],
                'internal': output['internal']
            }
    else:
        internal_text = output['internal'][-1][0]
        visible_text = output['visible'][-1][0]
        if regenerate:
            if loading_message:
                yield {
                     'visible': output[ 'visible'][:-1] + [[visible_text, shared.processing_message]],
                    'internal': output['internal'][:-1] + [[internal_text, '']]
                }
        elif _continue:
            last_reply = [output['internal'][-1][1], 
                          output[ 'visible'][-1][1]]
            if loading_message:
                yield {
                     'visible': output['visible'][:-1] + [[visible_text, last_reply[1] + '...']],
                    'internal': output['internal']
                }

    # Generate the prompt
    kwargs = {
        '_continue': _continue,
        'history': output if _continue else {k: v[:-1] for k, v in output.items()}
    }
    prompt = apply_extensions('custom_generate_chat_prompt', text, state, **kwargs)
    if prompt is None:
        prompt = generate_chat_prompt(text, state, **kwargs)

    # Generate
    generated = generate_reply(prompt, state, stopping_strings=stopping_strings, is_chat=True, for_ui=for_ui)

    for j, reply in enumerate(generated):

        # Extract the reply
        if state['mode'] in ['chat', 'chat-instruct']:
            visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply + '▍')
        else:
            visible_reply = reply + '▍'

        visible_reply = html.escape(visible_reply)

        if shared.stop_everything:
            if output['visible'][-1][1].endswith('▍'):
                output['visible'][-1][1] = output['visible'][-1][1][:-1]

            output['visible'][-1][1] = apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
            yield output
            return

        if _continue:
            output['internal'][-1] = [text, last_reply[0] + reply]
            output['visible'][-1] = [visible_text, last_reply[1] + visible_reply]
            if is_stream:
                yield output

        elif not (j == 0 and visible_reply.strip() == ''):
            output['internal'][-1] = [text, reply.lstrip(' ')]
            output['visible'][-1] = [visible_text, visible_reply.lstrip(' ')]
            if is_stream:
                yield output

    if output['visible'][-1][1].endswith('▍'):
        output['visible'][-1][1] = output['visible'][-1][1][:-1]
    output['visible'][-1][1] = apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
    yield output


def impersonate_wrapper(text, state):
    static_output = chat_html_wrapper(
        *[state[x] for x in ['history','name1','name2','mode','chat_style','character_menu']]
    )

    prompt = generate_chat_prompt('', state, impersonate=True)
    stopping_strings = get_stopping_strings(state)

    yield text + '...', static_output

    generated = generate_reply(prompt + text, state, stopping_strings=stopping_strings, is_chat=True)
    for reply in generated:
        yield (text + reply).lstrip(' '), 

        if shared.stop_everything:
            return


def generate_chat_reply_wrapper(text, state, regenerate=False, _continue=False):
    '''
    Same as above but returns HTML for the UI
    '''
    if not character_is_loaded(state):
        return

    if state['start_with'] != '' and not _continue:
        if regenerate:
            text, state['history'] = remove_last_message(state['history'])
            regenerate = False

        _continue = True
        send_dummy_message(text, state)
        send_dummy_reply(state['start_with'], state)

    history = state['history']

    generated = generate_chat_reply(text, state, regenerate, _continue, loading_message=True, for_ui=True)
    for i, history in enumerate(generated):
        yield chat_html_wrapper(
            history, *[state[x] for x in ['name1','name2','mode','chat_style','character_menu']]
        ), history

    save_history(history, state['unique_id'], state['character_menu'], state['mode'])


#########################################################
#               Handlers for Gradio-UI                  #
#########################################################

history_variables = ['unique_id','character_menu','mode']
redraw_variables = ['name1','name2','mode','chat_style','character_menu']


def handle_replace_last_reply_click(text, state):
    history = replace_last_reply(text, state)
    save_history(history, *[state[x] for x in history_variables])
    html = redraw(history, *[state[x] for x in redraw_variables])
    return [history, html, ""]


def handle_send_dummy_message_click(text, state):
    history = send_dummy_message(text, state)
    save_history(history, *[state[x] for x in history_variables])
    html = redraw(history, *[state[x] for x in redraw_variables])
    return [history, html, ""]


def handle_send_dummy_reply_click(text, state):
    history = send_dummy_reply(text, state)
    save_history(history, *[state[x] for x in history_variables])
    html = redraw(history, *[state[x] for x in redraw_variables])
    return [history, html, ""]


def handle_remove_last_click(state):
    last_input, history = remove_last_message(state['history'])
    save_history(history, *[state[x] for x in history_variables])
    html = redraw(history, *[state[x] for x in redraw_variables])
    return [history, html, last_input]


def handle_unique_id_select(state):
    history = load_history(*[state[x] for x in history_variables])
    html = redraw(history, *[state[x] for x in redraw_variables])
    convert_to_markdown.cache_clear()
    return [history, html]


def handle_start_new_chat_click(state):
    history = start_new_chat(state)
    html = redraw(history, *[state[x] for x in redraw_variables])

    histories = find_all_histories_with_first_prompts(state)
    convert_to_markdown.cache_clear()

    if len(histories) > 0:
        past_chats_update = gr.update(choices=histories, value=histories[0][1])
    else:
        past_chats_update = gr.update(choices=histories)

    return [history, html, past_chats_update]


def handle_delete_chat_confirm_click(state):
    index = str(find_all_histories(state).index(state['unique_id']))
    delete_history(*[state[x] for x in history_variables])

    history, unique_id = load_history_after_deletion(state, index)
    html = redraw(history, *[state[x] for x in redraw_variables])

    convert_to_markdown.cache_clear()
    return [
        history,
        html,
        unique_id,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False)
    ]


def handle_branch_chat_click(state):
    history = state['history']
    new_unique_id = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    save_history(history, new_unique_id, state['character_menu'], state['mode'])

    histories = find_all_histories_with_first_prompts(state)
    html = redraw(history, *[state[x] for x in redraw_variables])

    convert_to_markdown.cache_clear()
    return [history, html, gr.update(choices=histories, value=new_unique_id)]


def handle_rename_chat_click():
    return [
        gr.update(value="My New Chat"),
        gr.update(visible=True),
    ]


def handle_rename_chat_confirm(rename_to, state):
    rename_history(state['unique_id'], rename_to, state['character_menu'], state['mode'])
    histories = find_all_histories_with_first_prompts(state)

    return [
        gr.update(choices=histories, value=rename_to),
        gr.update(visible=False),
    ]


def handle_search_chat_change(state):
    histories = find_all_histories_with_first_prompts(state)
    return gr.update(choices=histories)


def handle_character_menu_change(state):
    name1, name2, picture, greeting, context = load_character(state['character_menu'], state['name1'], state['name2'])

    state['name1'] = name1
    state['name2'] = name2
    state['character_picture'] = picture
    state['greeting'] = greeting
    state['context'] = context

    history = load_latest_history(state)
    html = redraw(history, *[state[x] for x in redraw_variables])

    convert_to_markdown.cache_clear()

    histories = find_all_histories_with_first_prompts(state)
    if len(histories) > 0:
        past_chats_update = gr.update(choices=histories, value=histories[0][1])
    else:
        past_chats_update = gr.update(choices=histories)

    return [history, html, name1, name2, picture, greeting, context, past_chats_update]


def handle_mode_change(state):
    history = load_latest_history(state)
    histories = find_all_histories_with_first_prompts(state)
    html = redraw(history, *[state[x] for x in redraw_variables])

    convert_to_markdown.cache_clear()

    if len(histories) > 0:
        past_chats_update = gr.update(choices=histories, value=histories[0][1])
    else:
        past_chats_update = gr.update(choices=histories)

    return [
        history,
        html,
        gr.update(visible=state['mode'] != 'instruct'),
        gr.update(visible=state['mode'] == 'chat-instruct'),
        past_chats_update
    ]


def handle_save_character_click(name2):
    return [name2, gr.update(visible=True)]


def handle_load_template_click(instruction_template):
    output = load_instruction_template(instruction_template)
    return [output, "Select template to load..."]


def handle_save_template_click(instruction_template_str):
    contents = generate_instruction_template(instruction_template_str)
    return ["My Template.yaml", str(template_dir), contents, gr.update(visible=True)]


def handle_delete_template_click(template):
    return [f"{template}.yaml", str(template_dir), gr.update(visible=False)]


def handle_profile_change(picture, state):
    upload_profile_picture(picture)
    html = redraw(state['history'], *[state[x] for x in redraw_variables], reset_cache=True)
    return html


def handle_send_instruction_click(state):
    state['mode'] = 'instruct'
    state['history'] = {'internal': [], 'visible': []}
    output = generate_chat_prompt("Input", state)
    return output


def handle_send_chat_click(state):
    output = generate_chat_prompt("", state, _continue=True)
    return output


