import os
import sys
import importlib
import functools

import logging
import atexit
from termcolor import colored


def instantiate_from_config(config, **additional_kwargs):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")

    additional_kwargs.update(config.get("kwargs", dict()))
    return get_obj_from_str(config["target"])(**additional_kwargs)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


# Logger utils are copied from detectron2
class ColorfulFormatter(logging.Formatter):

    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # use 1K buffer if writing to cloud storage
    io = open(filename, "a", buffering=1024 if "://" in filename else -1)
    atexit.register(io.close)
    return io


@functools.lru_cache()
def setup_logger(output, distributed_rank, color=True, name='AnimateDiff', abbrev_name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = 'AD'
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s:%(lineno)d %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = ColorfulFormatter(
                colored("[%(asctime)s %(name)s:%(lineno)d]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


def format_time(elapsed_time):
    # Time thresholds
    minute = 60
    hour = 60 * minute
    day = 24 * hour

    days, remainder = divmod(elapsed_time, day)
    hours, remainder = divmod(remainder, hour)
    minutes, seconds = divmod(remainder, minute)

    formatted_time = ""

    if days > 0:
        formatted_time += f"{int(days)} days "
    if hours > 0:
        formatted_time += f"{int(hours)} hours "
    if minutes > 0:
        formatted_time += f"{int(minutes)} minutes "
    if seconds > 0:
        formatted_time += f"{seconds:.2f} seconds"

    return formatted_time.strip()

