import torch
from transformers import AutoConfig, AutoTokenizer

from ... import shared
from ...path import MLM_LOCAL_MODELS


def load_model(model_name: str):
    assert model_name in MLM_LOCAL_MODELS
    checkpoint_dir = MLM_LOCAL_MODELS[model_name]

    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=shared.model_args.trust_remote_code)
    
    if config.to_dict().get('is_encoder_decoder', False):
        from transformers import AutoModelForSeq2SeqLM as AutoGenerator
        shared.is_seq2seq = True

    elif 'chatglm' in model_name.lower():
        from transformers import AutoModel as AutoGenerator

    else:
        from transformers import AutoModelForCausalLM as AutoGenerator

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    generator = AutoGenerator.from_pretrained(checkpoint_dir, torch_dtype='auto', 
                                                        low_cpu_mem_usage=shared.model_args.low_cpu_mem)
    generator = generator.to(device=shared.device)
    generator.eval()
    return tokenizer, generator


build_prompt = None


def generate_response(
    generator,
    tokenizer,
    prompt,
    **kwargs
):
    with torch.no_grad():
        tokens = tokenizer(prompt, return_tensors="pt").input_ids
        tokens = tokens.to(device=shared.device, non_blocking=True)

        output = generator.generate(tokens, **kwargs)[0]
        output = tokenizer.decode(output, skip_special_tokens=True)

    return output

