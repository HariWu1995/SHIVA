import os
import json
from pathlib import Path
from ..utils import natural_keys


ROOT_DIR = Path(__file__).resolve().parent
PROMPT_DIR = ROOT_DIR / 'prompts'


def get_available_prompts():
    paths = list(PROMPT_DIR.glob('*.txt'))
    paths = sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)
    prompts = [k.stem for k in paths]
    return prompts


def load_prompt(name):
    path = PROMPT_DIR / f'{name}.txt'
    with open(path, 'r') as file:
        data = file.read()
    return data


if __name__ == "__main__":

    all_prompts = get_available_prompts()
    print(json.dumps(all_prompts, indent=4))

    print("\n\n")
    prompt_name = all_prompts[0]
    prompt = load_prompt(prompt_name)
    print(prompt_name)
    print(prompt)

