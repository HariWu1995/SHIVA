import json
import shutil
from pathlib import Path
from PIL import Image

from ...grui.html import make_thumbnail

from .. import shared
from ..logging import logger
from ..utils import get_available_characters

from .utils import yaml


CHARACTER_DIR = Path(__file__).resolve().parents[5] / "logs/chat/characters"
if CHARACTER_DIR.exists() is False:
    CHARACTER_DIR.mkdir()

default_character_dir = Path(__file__).resolve().parents[1] / "characters"
shutil.copytree(str(default_character_dir), str(CHARACTER_DIR), dirs_exist_ok=True)

cache_folder = Path(shared.args.disk_cache_dir)
if cache_folder.exists() is False:
    cache_folder.mkdir()


def generate_pfp_cache(character):

    for path in [CHARACTER_DIR / f"{character}.{ext}" for ext in ['png', 'jpg', 'jpeg']]:
        if path.exists() is False:
            continue

        img = Image.open(path)
        img.save(cache_folder / 'pfp_character.png', format='PNG')

        thumb = make_thumbnail(img)
        thumb.save(cache_folder / 'pfp_character_thumb.png', format='PNG')

        return thumb

    return None


def load_character(character, name1, name2):
    context = ""
    greeting = ""
    greeting_field = 'greeting'
    picture = None

    filepath = None
    for ext in ["yml", "yaml", "json"]:
        filepath = CHARACTER_DIR / f"{character}.{ext}"
        if filepath.exists():
            break

    if filepath is None or not filepath.exists():
        logger.error(f"Could not find the character \"{character}\" inside characters/. No character has been loaded.")
        raise ValueError

    file_contents = open(filepath, 'r', encoding='utf-8').read()
    if ext == "json":
        data = json.loads(file_contents)
    else:
        data = yaml.safe_load(file_contents)

    # Delete old files before create new ones
    for path in [cache_folder / 'pfp_character.png', 
                 cache_folder / 'pfp_character_thumb.png']:
        if path.exists():
            path.unlink()

    picture = generate_pfp_cache(character)

    # Finding the bot's name
    for k in ['name', 'bot', '<|bot|>', 'char_name']:
        if k in data and data[k] != '':
            name2 = data[k]
            break

    # Find the user name (if any)
    for k in ['your_name', 'user', '<|user|>']:
        if k in data and data[k] != '':
            name1 = data[k]
            break

    if 'context' in data:
        context = data['context'].strip()

    elif "char_persona" in data:
        context = build_pygmalion_style_context(data)
        greeting_field = 'char_greeting'

    greeting = data.get(greeting_field, greeting)
    return name1, name2, picture, greeting, context


def upload_character(file, img, tavern=False):
    decoded_file = file if isinstance(file, str) else file.decode('utf-8')
    try:
        data = json.loads(decoded_file)
    except:
        data = yaml.safe_load(decoded_file)

    if 'char_name' in data:
        name = data['char_name']
        greeting = data['char_greeting']
        context = build_pygmalion_style_context(data)
        yaml_data = generate_character_yaml(name, greeting, context)
    else:
        name = data['name']
        yaml_data = generate_character_yaml(data['name'], data['greeting'], data['context'])

    out_filename = name
    i = 1
    while (CHARACTER_DIR / f'{out_filename}.yaml').exists():
        out_filename = f'{name}_{i:03d}'
        i += 1

    with open(CHARACTER_DIR / f'{out_filename}.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml_data)

    if img is not None:
        img.save(CHARACTER_DIR / f'{out_filename}.png')

    out_filepath = str(CHARACTER_DIR / f'{out_filename}.yaml')
    logger.info(f'New character saved to `{out_filepath}`.')
    return out_filename
    # return gr.update(value=out_filename, choices=get_available_characters())


def upload_tavern_character(img, _json):
    _json = {
        'char_name': _json['name'], 
        'char_persona': _json['description'], 
        'char_greeting': _json['first_mes'], 
        'example_dialogue': _json['mes_example'], 
        'world_scenario': _json['scenario'],
    }
    return upload_character(json.dumps(_json), img, tavern=True)


def check_tavern_character(img):
    if "chara" not in img.info:
        return "Not a TavernAI card", None, None

    decoded_string = base64.b64decode(img.info['chara']).replace(b'\\r\\n', b'\\n')
    _json = json.loads(decoded_string)
    if "data" in _json:
        _json = _json["data"]
    return _json['name'], _json['description'], _json


def generate_character_yaml(name, greeting, context):
    data = {
        'name': name,
        'greeting': greeting,
        'context': context,
    }
    data = {k: v for k, v in data.items() if v}  # Strip falsy
    return yaml.dump(data, sort_keys=False, width=float("inf"))


def save_character(name, greeting, context, picture = None, filename: str = ''):
    if filename == "":
        logger.error("The filename is empty, so the character will not be saved.")
        return

    data = generate_character_yaml(name, greeting, context)

    filepath = CHARACTER_DIR / f'{filename}.yaml'
    save_file(filepath, data)

    image_path = CHARACTER_DIR / f'{filename}.png'
    if picture is not None:
        picture.save(image_path)
        logger.info(f'Saved {image_path}.')


def delete_character(name, instruct=False):
    for ext in ["yml", "yaml", "json", "png"]:
        delete_file(CHARACTER_DIR / f'{name}.{ext}')


def update_character_menu_after_deletion(idx):
    characters = get_available_characters()
    idx = min(int(idx), len(characters) - 1)
    idx = max(0, idx)
    return characters, characters[idx]
    # return gr.update(choices=characters, value=characters[idx])


def upload_profile_picture(img):
    path_to_profile_pic = cache_folder / "pfp_me.png"
    if img is None:
        return
    img = make_thumbnail(img)
    img.save(path_to_profile_pic)
    logger.info(f'Profile picture saved to "{str(path_to_profile_pic)}"')


def build_pygmalion_style_context(data):
    context = ""
    if 'char_persona' in data and data['char_persona'] != '':
        context += f"{data['char_name']}'s Persona: {data['char_persona']}\n"

    if 'world_scenario' in data and data['world_scenario'] != '':
        context += f"Scenario: {data['world_scenario']}\n"

    if 'example_dialogue' in data and data['example_dialogue'] != '':
        context += f"{data['example_dialogue'].strip()}\n"

    context = f"{context.strip()}\n"
    return context

