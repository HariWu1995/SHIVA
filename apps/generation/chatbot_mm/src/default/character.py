import os
import json
import yaml
from pathlib import Path
from ..utils import natural_keys


ROOT_DIR = Path(__file__).resolve().parent
CHARACTER_DIR = ROOT_DIR / 'characters'


def get_available_characters():
    paths = [x for x in CHARACTER_DIR.iterdir() if x.suffix in ('.json', '.yaml', '.yml')]
    return sorted(set([k.stem for k in paths]), key=natural_keys)


def load_character(name):
    paths = CHARACTER_DIR.glob(f'{name}.*')
    path = sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    ext = os.path.splitext(path)[-1].lower()
    assert ext in ['.json', '.yaml', '.yml']
    with open(path, 'r') as file:
        if ext == ".json":
            data = json.load(file)
        else:
            data = yaml.load(file, Loader=yaml.SafeLoader)
    return data


if __name__ == "__main__":

    all_characters = get_available_characters()
    print(json.dumps(all_characters, indent=4))

    print("\n\n")
    char_name = all_characters[0]
    char_dict = load_character(char_name)
    print(char_name)
    print(json.dumps(char_dict, indent=4))

