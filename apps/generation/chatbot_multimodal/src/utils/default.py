from pathlib import Path

from .language import natural_keys


ROOT_DIR = Path(__file__).resolve().parents[1]

INSTRUCTION_DIR = ROOT_DIR / 'instructions'
CHARACTER_DIR = ROOT_DIR / 'characters'
PRESET_DIR = ROOT_DIR / 'presets'
PROMPT_DIR = ROOT_DIR / 'prompts'
GRAMMAR_DIR = ROOT_DIR / 'grammar/grammars'


def get_available_characters():
    paths = [
        x for x in CHARACTER_DIR.iterdir() 
           if x.suffix in ('.json', '.yaml', '.yml')
    ]
    return sorted(set([k.stem for k in paths]), key=natural_keys)


def get_available_presets():
    paths = [k.stem for k in PRESET_DIR.glob('*.yaml')]
    return sorted(set(paths), key=natural_keys)


def get_available_prompts():
    prompt_files = list(PROMPT_DIR.glob('*.txt'))
    sorted_files = sorted(prompt_files, key=lambda x: x.stat().st_mtime, reverse=True)
    prompts = [file.stem for file in sorted_files]
    prompts.append('None')
    return prompts


def get_available_instructions():
    paths = [
        x for x in INSTRUCTION_DIR.iterdir() 
           if x.suffix in ('.json', '.yaml', '.yml')
    ]
    return ['None'] + sorted(set((k.stem for k in paths)), key=natural_keys)


def get_available_grammars():
    paths = [x for x in GRAMMAR_DIR.glob('*.gbnf')]
    return ['None'] + sorted([item.name for item in paths], key=natural_keys)


def get_available_chat_styles():
    UI_CSS_DIR = ROOT_DIR.parent / "grui/css"
    paths = ['-'.join(k.stem.split('-')[1:]) 
                  for k in UI_CSS_DIR.glob('chat_style*.css')]
    return sorted(set(paths), key=natural_keys)
