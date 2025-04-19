import os
import re
import html
import json
from datetime import datetime
from pathlib import Path

from ...grui import shared as ui
from ..logging import logger
from ..utils import save_file, delete_file
from .utils import replace_character_names, redraw_html


HISTORY_DIR = Path(__file__).resolve().parents[5] / "logs/chat/history"

HISTORY_INSTRUCT_DIR  = HISTORY_DIR / 'instruct'
HISTORY_CHARACTER_DIR = HISTORY_DIR / 'character'

for subfolder_path in [HISTORY_INSTRUCT_DIR, HISTORY_CHARACTER_DIR]:
    if not subfolder_path.exists():
        os.makedirs(subfolder_path)


def delete_history(unique_id, character, mode):
    p = get_history_file_path(unique_id, character, mode)
    delete_file(p)


def save_history(history, unique_id, character, mode):
    if ui.multi_user:
        return

    p = get_history_file_path(unique_id, character, mode)
    if p.parent.is_dir() is False:
        p.parent.mkdir(parents=True)

    with open(p, mode='w', encoding='utf-8') as f:
        f.write(json.dumps(history, indent=4, ensure_ascii=False))


def rename_history(old_id, new_id, character, mode):
    if ui.multi_user:
        return

    old_p = get_history_file_path(old_id, character, mode)
    new_p = get_history_file_path(new_id, character, mode)

    if new_p.parent != old_p.parent:
        logger.error(f"The following path is not allowed: \"{new_p}\".")
    elif new_p == old_p:
        logger.info("The provided path is identical to the old one.")
    elif new_p.exists():
        logger.error(f"The new path already exists and will not be overwritten: \"{new_p}\".")
    else:
        logger.info(f"Renaming \"{old_p}\" to \"{new_p}\"")
        old_p.rename(new_p)


def get_history_file_path(unique_id, character: str = '', mode: str = 'instruct'):
    if mode == 'instruct':
        fpath = HISTORY_INSTRUCT_DIR / f'{unique_id}.json'
    else:
        fpath = HISTORY_CHARACTER_DIR / f'{character}/{unique_id}.json'
    return fpath


def get_all_paths(state):

    if state['mode'] != 'instruct':
        character = state['character_menu']

        # Handle obsolete filenames and paths
        old_p = HISTORY_DIR / f'{character}_persistent.json'
        new_p = HISTORY_DIR / f'persistent_{character}.json'

        if old_p.exists():
            logger.warning(f"Renaming \"{old_p}\" to \"{new_p}\"")
            old_p.rename(new_p)

        if new_p.exists():
            unique_id = datetime.now().strftime('%Y%m%d-%H-%M-%S')
            p = get_history_file_path(unique_id, character, state['mode'])
            logger.warning(f"Moving \"{new_p}\" to \"{p}\"")
            p.parent.mkdir(exist_ok=True)
            new_p.rename(p)

        return (HISTORY_CHARACTER_DIR / character).glob('*.json')

    else:
        return HISTORY_INSTRUCT_DIR.glob('*.json')


def find_all_histories(state):
    if ui.multi_user:
        return ['']

    paths = get_all_paths(state)
    histories = sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)
    return [path.stem for path in histories]


def find_all_histories_with_first_prompts(state):
    if ui.multi_user:
        return []

    paths = get_all_paths(state)
    histories = sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)

    result = []
    for i, path in enumerate(histories):
        filename = path.stem
        file_content = ""
        with open(path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        if state['search_chat'] \
        and state['search_chat'] not in file_content:
            continue

        dt_format = r'^[0-9]{8}-[0-9]{2}-[0-9]{2}-[0-9]{2}$'
        check_fn_as_dt_format = re.match(dt_format, filename)

        if check_fn_as_dt_format:
            data = json.loads(file_content)
            first_prompt = ""
            if data and 'visible' in data and len(data['visible']) > 0:
                if data['internal'][0][0] == '<|BEGIN-VISIBLE-CHAT|>':
                    if len(data['visible']) > 1:
                        first_prompt = html.unescape(data['visible'][1][0])
                    elif i == 0:
                        first_prompt = "New chat"
                else:
                    first_prompt = html.unescape(data['visible'][0][0])
            elif i == 0:
                first_prompt = "New chat"
        else:
            first_prompt = filename
        first_prompt = first_prompt.strip()

        # Truncate the first prompt if it's longer than 30 characters
        if len(first_prompt) > 30:
            first_prompt = first_prompt[:30 - 3] + '...'

        result.append((first_prompt, filename))

    return result


def load_latest_history(state):
    '''
    Loads the latest history for the given character in chat or chat-instruct mode,
    or the latest instruct history for instruct mode.
    '''
    if ui.multi_user:
        return start_new_chat(state)

    histories = find_all_histories(state)
    if len(histories) > 0:
        history = load_history(histories[0], state['character_menu'], state['mode'])
    else:
        history = start_new_chat(state)
    return history


def load_history_after_deletion(state, idx):
    '''
    Loads the latest history for the given character in chat or chat-instruct mode,
    or the latest instruct history for instruct mode.
    '''
    if ui.multi_user:
        return start_new_chat(state)

    histories = find_all_histories_with_first_prompts(state)
    idx = min(int(idx), len(histories) - 1)
    idx = max(0, idx)

    if len(histories) > 0:
        history = load_history(histories[idx][1], state['character_menu'], state['mode'])
    else:
        history = start_new_chat(state)
        histories = find_all_histories_with_first_prompts(state)
    return history, histories, idx
    # return history, gr.update(choices=histories, value=histories[idx][1])


def start_new_chat(state):
    mode = state['mode']
    history = {
        'internal': [], 
        'visible': [],
    }

    if mode != 'instruct':
        greeting = replace_character_names(state['greeting'], state['name1'], state['name2'])
        if greeting != '':
            history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', greeting]]
            history['visible'] += [['', html.escape(greeting)]]

    unique_id = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    save_history(history, unique_id, state['character_menu'], state['mode'])
    return history


def load_history(unique_id, character, mode):
    p = get_history_file_path(unique_id, character, mode)

    f = json.loads(open(p, 'rb').read())
    if 'internal' in f \
    and 'visible' in f:
        history = f
    else:
        history = {'internal': f['data'], 'visible': f['data_visible']}
    return history


def load_history_json(file, history):
    try:
        file = file.decode('utf-8')
        f = json.loads(file)
        if 'internal' in f \
        and 'visible' in f:
            history = f
        else:
            history = {'internal': f['data'], 'visible': f['data_visible']}
        return history
    except:
        return history


def handle_upload_chat_history(load_chat_history, state):
    history = start_new_chat(state)
    history = load_history_json(load_chat_history, history)

    save_history(history, state['unique_id'], state['character_menu'], state['mode'])
    histories = find_all_histories_with_first_prompts(state)

    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    from ..grui.html import convert_to_markdown
    convert_to_markdown.cache_clear()

    if len(histories) > 0:
        past_chats_update = gr.update(choices=histories, value=histories[0][1])
    else:
        past_chats_update = gr.update(choices=histories)

    return [
        history,
        html,
        past_chats_update
    ]

