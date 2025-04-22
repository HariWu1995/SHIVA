import os
import json
import yaml
from pathlib import Path
from ..utils import natural_keys


ROOT_DIR = Path(__file__).resolve().parent
INSTRUCTION_DIR = ROOT_DIR / 'instructions'


def get_available_instructions():
    paths = [x for x in INSTRUCTION_DIR.iterdir() if x.suffix in ('.json', '.yaml', '.yml')]
    return sorted(set((k.stem for k in paths)), key=natural_keys)


def load_instruction(name):
    paths = INSTRUCTION_DIR.glob(f'{name}.*')
    path = sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    ext = os.path.splitext(path)[-1].lower()
    assert ext in ['.json', '.yaml', '.yml']
    with open(path, 'r') as file:
        if ext == ".json":
            data = json.load(file)
        else:
            data = yaml.load(file, Loader=yaml.SafeLoader)
    return data


def generate_instruction(instruction_template):
    data = dict(instruction_template=instruction_template)
    result = ""
    for k in data:
        result += k + ": |-\n"
        for line in data[k].splitlines():
            result += "  " + line.rstrip(' ') + "\n"
    return result


def save_instruction(model, template):
    if model == 'None':
        yield ("Not saving the template because no model is selected in the menu.")
        return

    from .. import shared
    user_config = shared.user_config

    model_regex = model + '$'  # For exact matches
    if model_regex not in user_config:
        user_config[model_regex] = {}

    if template == 'None':
        user_config[model_regex].pop('instruction_template', None)
    else:
        user_config[model_regex]['instruction_template'] = template

    shared.user_config = user_config
    uconfig = yaml.dump(user_config, sort_keys=False)

    path = str(shared.user_config_path)
    with open(path, 'w') as f:
        f.write(uconfig)

    if template == 'None':
        yield (f"Instruction template for `{model}` unset in `{path}`, as the value for template was `{template}`.")
    else:
        yield (f"Instruction template for `{model}` saved to `{path}` as `{template}`.")


if __name__ == "__main__":

    all_instructions = get_available_instructions()
    print(json.dumps(all_instructions, indent=4))

    print("\n\n")
    instr_name = all_instructions[0]
    instr_dict = load_instruction(instr_name)
    print(instr_name)
    print(instr_dict["instruction_template"])

