import yaml
from types import SimpleNamespace as Namespace
from pathlib import Path, PurePath
from collections import OrderedDict


def load_config(config_path: str | PurePath, return_type: str = "namespace"):
    if not isinstance(config_path, PurePath):
        config_path = Path(config_path)
    if config_path.exists():
        config = yaml.safe_load(open(config_path, 'r').read())
    else:
        config = {}
    return_type = return_type.lower()
    if return_type == "namespace":
        return Namespace(**config)
    else:
        return OrderedDict(config)


