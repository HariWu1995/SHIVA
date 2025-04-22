import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration as QVenGenerator
from qwen_vl_utils import process_vision_info

from ... import shared
from ...path import MLM_LOCAL_MODELS


generation_config = dict(max_new_tokens=512)

image_extensions = ['.jpg', '.jpeg', '.png']


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


def build_prompt(
    content: str, 
    role: str = 'user', 
    chat_history: list = [],
):
    chat_history.append({"role": role, "content": content})
    return chat_history


def generate_response(
    generator,
    processor,
    text_prompt: str | None = None,
    chat_history: list | None = None,
    **kwargs
):
    # Preprocessing
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


if __name__ == "__main__":

    processor, generator = load_model("qwen-2.5VL-7b-instruct")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://storage.googleapis.com/digital-platform/hinh_anh_review_khac_san_landmark_81_tien_ich_and_gia_phong_moi_nhat_so_2_ce2f8922a5/hinh_anh_review_khac_san_landmark_81_tien_ich_and_gia_phong_moi_nhat_so_2_ce2f8922a5.jpg"},
                {"type": "image", "image": "https://static.vinwonders.com/production/2025/02/thong-tin-ve-landmark-81.jpg"},
                {"type": "text", "text": "Describe the similarities and differences between these images."},
            ],
        }
    ]

    response = generate_response(generator, processor, chat_history=messages, **generation_config)
    print(response)

