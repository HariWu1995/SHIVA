import os
import json
import yaml
from pathlib import Path
from ..utils import natural_keys


ROOT_DIR = Path(__file__).resolve().parent
GRAMMAR_DIR = ROOT_DIR / 'grammars'


def get_available_grammars():
    paths = [x for x in GRAMMAR_DIR.glob('*.gbnf')]
    return sorted([k.stem for k in paths], key=natural_keys)


def load_grammar(name):
    path = GRAMMAR_DIR / f'{name}.gbnf'
    with open(path, 'r') as file:
        data = file.read()
    return data


if __name__ == "__main__":

    all_grammars = get_available_grammars()
    print(json.dumps(all_grammars, indent=4))

    print("\n\n")
    grmr_name = all_grammars[0]
    grammar = load_grammar(grmr_name)
    print(grmr_name)
    print(grammar)

