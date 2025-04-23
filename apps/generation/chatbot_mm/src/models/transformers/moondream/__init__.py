# from transformers import AutoModelForCausalLM as AutoGenerator
from .hf_moondream import HfMoondream as AutoGenerator

from PIL import Image
from .... import shared
from ....path import MLM_LOCAL_MODELS
from ....utils import get_file_type


generation_config = dict()


def load_model(model_name: str):
    assert model_name in MLM_LOCAL_MODELS
    checkpoint_dir = MLM_LOCAL_MODELS[model_name]

    generator = AutoGenerator.from_pretrained(checkpoint_dir, torch_dtype='auto', 
                                                        trust_remote_code=True,
                                                        low_cpu_mem_usage=shared.model_args.low_cpu_mem)
    generator = generator.to(device=shared.device)
    generator.eval()
    return None, generator


def build_prompt(chat_history: list = []):

    mm_prompt = dict()

    while not all([k in mm_prompt.keys() for k in ['image','prompt']]):
        message = chat_history.pop(-1)
        mc = message['content']

        if isinstance(mc, str):
            if 'prompt' not in mm_prompt.keys():
                mm_prompt['prompt'] = mc
            # else:
            #     mm_prompt['prompt'] = mc + ' ' + mm_prompt['prompt']

        elif isinstance(mc, (list, tuple)):
            mtype = get_file_type(mc[0])
            if mtype != 'image':
                continue
            mm_prompt['image'] = Image.open(mc[0])

    # TODO: use extra LLM to extract `task`
    raw_prompt = mm_prompt['prompt'].lower()
    if 'detect' in raw_prompt:
        mm_prompt['task'] = 'detect'
    elif 'focus' in raw_prompt:
        mm_prompt['task'] = 'focus'
    elif 'caption' in raw_prompt:
        if any([x in raw_prompt for x in ['brief','short']]):
            mm_prompt['task'] = 'caption-short'
        elif 'long' in raw_prompt:
            mm_prompt['task'] = 'caption-long'
        else:
            mm_prompt['task'] = 'caption-normal'
    elif 'describe' in raw_prompt or 'description' in raw_prompt:
        mm_prompt['task'] = 'query'
    return mm_prompt


def parse_response(response, role: str = 'assistant'):
    return {'role': role, 'content': str(response)}


def generate_response(
    generator,
    tokenizer = None,
    image: Image.Image | None = None,
    prompt: str = '',
    task: str = 'query',
    **kwargs
):
    assert image is not None
    assert task in ['query', 'detect', 'focus', 'caption-short', 'caption-long', 'caption-normal']
    if task == 'query':
        return query(generator, image, question=prompt)
    elif task == 'detect':
        return detect(generator, image, object_name=prompt)
    elif task == 'focus':
        return focus(generator, image, object_name=prompt)
    else:
        return caption(generator, image, length=task.split('-')[1])


def caption(generator, image: Image.Image, length: str = 'short'):
    if length not in ['short','normal','long']:
        length = 'normal'
    response = generator.caption(image, length=length)["caption"]
    return response


def detect(generator, image: Image.Image, object_name: str):
    response = generator.detect(image, object_name)["objects"]
    return response


def focus(generator, image: Image.Image, object_name: str):
    response = generator.point(image, object_name)["points"]
    return response


def query(generator, image: Image.Image, question: str):
    response = generator.query(image, question)["answer"]
    return response


if __name__ == "__main__":
    path = "C:/Users/Mr. RIAH/Pictures/_character/Nhi-04.jpg"
    image = Image.open(path)

    _, generator = load_model("moondream-v2")

    response = caption(generator, image, 'short')
    print("\n\n Captioning:", response)

    response = detect(generator, image, 'girl')
    print("\n\n Detection:", response)

    response = point(generator, image, 'girl')
    print("\n\n Pointing:", response)

    response = query(generator, image, 'describe the picture, focus on the emotion of people.')
    print("\n\n QnA:", response)

