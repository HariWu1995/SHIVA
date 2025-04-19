from pathlib import Path

from .generation import get_encoded_length


prompt_dir = Path(__file__).resolve().parent / "prompts"


def load_prompt(fname: str = ''):
    if fname in ['None', '']:
        return ''
    else:
        file_path = prompt_dir / f'{fname}.txt'
        if not file_path.exists():
            return ''

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if text[-1] == '\n':
            text = text[:-1]
        return text


def count_tokens(text):
    try:
        tokens = get_encoded_length(text)
        return str(tokens)
    except:
        return '0'
