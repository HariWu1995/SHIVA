# from transformers import AutoModelForCausalLM as AutoGenerator
from .hf_moondream import HfMoondream as AutoGenerator

from PIL import Image
from .... import shared
from ....path import MLM_LOCAL_MODELS


def load_model(model_name: str):
    assert model_name in MLM_LOCAL_MODELS
    checkpoint_dir = MLM_LOCAL_MODELS[model_name]

    generator = AutoGenerator.from_pretrained(checkpoint_dir, torch_dtype='auto', 
                                                        trust_remote_code=True,
                                                        low_cpu_mem_usage=shared.model_args.low_cpu_mem)
    generator = generator.to(device=shared.device, non_blocking=True)
    generator.eval()
    return None, generator


build_prompt = None


def generate_response(
    generator,
    image,
    prompt: str = '',
    task: str = 'query',
    **kwargs
):
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

