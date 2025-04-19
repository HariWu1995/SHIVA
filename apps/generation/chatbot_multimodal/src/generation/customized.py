import time
import traceback

from .. import shared
from ..logging import logger

from .utils import set_manual_seed, print_prompt


def generate_reply_custom(question, original_question, seed, state, stopping_strings=None, is_chat=False):
    """
    For models that do not use the transformers library for sampling
    """
    seed = set_manual_seed(state['seed'])

    t0 = time.time()
    reply = ''
    try:
        if not is_chat:
            yield ''

        if not state['stream']:
            reply = shared.model.generate(question, state)
            yield reply
        else:
            for reply in shared.model.generate_with_streaming(question, state):
                yield reply

    except Exception:
        traceback.print_exc()

    finally:
        t1 = time.time()
        original_tokens = len(encode(original_question)[0])
        new_tokens = len(encode(original_question + reply)[0]) - original_tokens
        logger.info(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return

