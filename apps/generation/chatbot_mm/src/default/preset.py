import os
import json
import yaml
from pathlib import Path
from ..utils import natural_keys


ROOT_DIR = Path(__file__).resolve().parent
PRESET_DIR = ROOT_DIR / 'presets'


def get_available_presets():
    paths = [k.stem for k in PRESET_DIR.glob('*.yaml')]
    return sorted(set(paths), key=natural_keys)


def load_preset(name):
    path = PRESET_DIR / f'{name}.yaml'
    with open(path, 'r') as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)
    return data


def default_preset():
    return {
        'temperature': 1,
        'temperature_last': False,
        'dynamic_temperature': False,
        'dynatemp_low': 1,
        'dynatemp_high': 1,
        'dynatemp_exponent': 1,
        'smoothing_factor': 0,
        'smoothing_curve': 1,
        'min_p': 0,
        'top_p': 1,
        'top_k': 0,
        'typical_p': 1,
        'xtc_threshold': 0.1,
        'xtc_probability': 0,
        'eps_cutoff': 0,
        'eta_cutoff': 0,
        'tfs': 1,
        'top_a': 0,
        'top_n_sigma': 0,
        'dry_base': 1.75,
        'dry_multiplier': 0,
        'dry_length': 2,
        'dry_sequence_breakers': '"\\n", ":", "\\"", "*"',
        'penalty_alpha': 0,
        'frequency_penalty': 0,
        'presence_penalty': 0,
        'encoder_repetition_penalty': 1,
        'repetition_penalty': 1,
        'repetition_penalty_range': 1024,
        'no_repeat_ngram_size': 0,
        'guidance_scale': 1,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'do_sample': True,
        'sampler_priority': [
            'repetition_penalty',
            'presence_penalty',
            'frequency_penalty',
            'dry',
            'temperature',
            'dynamic_temperature',
            'quadratic_sampling',
            'top_n_sigma',
            'top_k',
            'top_p',
            'typical_p',
            'eps_cutoff',
            'eta_cutoff',
            'tfs',
            'top_a',
            'min_p',
            'mirostat',
            'xtc',
            'encoder_repetition_penalty',
            'no_repeat_ngram'
        ],
    }


if __name__ == "__main__":

    all_presets = get_available_presets()
    print(json.dumps(all_presets, indent=4))

    print("\n\n")
    preset_name = all_presets[0]
    preset = load_preset(preset_name)
    print(preset_name)
    print(json.dumps(preset, indent=4))

