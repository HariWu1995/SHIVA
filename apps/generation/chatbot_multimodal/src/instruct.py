import yaml
from pathlib import Path

template_dir = Path(__file__).resolve().parent / "instructions"


def load_instruction_template(template):
    if template == 'None':
        return ''

    template_path = template_dir / f"{template}.yaml"
    if not template_path.exists():
        return ''

    content = open(template_path, 'r', encoding='utf-8').read()
    data = yaml.safe_load(content)
    if 'instruction_template' in data:
        return data['instruction_template']
    return ''
    # return jinja_template_from_old_format(data)


def generate_instruction_template(instruction_template):
    data = {'instruction_template': instruction_template}
    return format_yaml(data)


def format_yaml(data):
    '''
    pyyaml is very inconsistent with multiline strings.
    for simple instruction template outputs, this is enough.
    '''
    result = ""
    for k in data:
        result += k + ": |-\n"
        for line in data[k].splitlines():
            result += "  " + line.rstrip(' ') + "\n"
    return result

