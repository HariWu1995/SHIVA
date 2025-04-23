import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration as QVenGenerator
from qwen_vl_utils import process_vision_info

from ... import shared
from ...path import MLM_LOCAL_MODELS
from ...utils import get_file_type
from ...logging import logger


generation_config = dict(max_new_tokens=512)


def load_model(model_name: str):
    assert model_name in MLM_LOCAL_MODELS
    checkpoint_dir = MLM_LOCAL_MODELS[model_name]

    min_pixels =  256 * 28 * 28
    max_pixels = 1280 * 28 * 28

    processor = AutoProcessor.from_pretrained(checkpoint_dir, min_pixels=min_pixels, max_pixels=max_pixels)
    generator = AutoGenerator.from_pretrained(checkpoint_dir, torch_dtype='auto', 
                                                        # attn_implementation="flash_attention_2",
                                                        low_cpu_mem_usage=shared.model_args.low_cpu_mem)
    generator = generator.to(device=shared.device, non_blocking=True)
    generator.eval()
    return processor, generator


def build_prompt(chat_history: list = []):
    """
    Message Format:
        +   text: {'role': 'user', 'content':  text  , 'metadata': None, 'options': None}
        + others: {'role': 'user', 'content': (path,), 'metadata': None, 'options': None}
    """
    prompted_history = []

    for m in chat_history:
        mc = m['content']

        if isinstance(mc, str):
            mtype = 'text'
            mcontent = mc

        elif isinstance(mc, (list, tuple)):
            mtype = get_file_type(mc[0])
            if mtype in ['image','video']:
                mcontent = mc[0]
            else:
                logger.warn(f"{mc[0]} (with type = {mtype}) will be ignored!")
                continue
        else:
            logger.warn(f"{mc} cannot be parsed and will be ignored!")
            continue

        prompted_history.append({
                "role": m['role'], 
             "content": [{"type": mtype, mtype: mcontent}]
        })
        del mcontent, mtype

    return dict(chat_history=prompted_history)


def generate_response(
    generator,
    processor,
    text_prompt: str | None = None,
    chat_history: list | None = None,
    **kwargs
):
    # Preprocessing: https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal
    if text_prompt is None:
        text_prompt = processor.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    image_inputs, \
    video_inputs = process_vision_info(chat_history)

    # Generation
    with torch.no_grad():
        inputs = processor(text=[text_prompt], 
                         images=image_inputs, padding=True,
                         videos=video_inputs, return_tensors="pt")
        inputs = inputs.to(device=shared.device)

        output = generator.generate(**inputs, **kwargs)[0]
        output = tokenizer.decode(output, skip_special_tokens=True, 
                                  clean_up_tokenization_spaces=False)
    return output


def parse_response(output, role: str = 'assistant')
    return {'role': role, 'content': output}


if __name__ == "__main__":

    processor, generator = load_model("qwen-2.5VL-7b-instruct")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://storage.googleapis.com/digital-platform/hinh_anh_review_khac_san_landmark_81_tien_ich_and_gia_phong_moi_nhat_so_2_ce2f8922a5/hinh_anh_review_khac_san_landmark_81_tien_ich_and_gia_phong_moi_nhat_so_2_ce2f8922a5.jpg"},
                {"type": "image", "image": "https://static.vinwonders.com/production/2025/02/thong-tin-ve-landmark-81.jpg"},
                {"type":  "text",  "text": "Describe the similarities and differences between these images."},
            ],
        }
    ]

    response = generate_response(generator, processor, chat_history=messages, **generation_config)
    print(response)

