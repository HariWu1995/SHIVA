"""
Evaluation: 
    - work in question-and-answer manner
    - work not well in dialogue
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM as AutoGenerator

from ... import shared
from ...path import MLM_LOCAL_MODELS


INSTRUCTION = "Given a dialog context and related knowledge, you need to response naturally based on the knowledge."


def load_model(model_name: str = "godel-1.1large-chat"):
    assert model_name in MLM_LOCAL_MODELS
    checkpoint_dir = MLM_LOCAL_MODELS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    generator = AutoGenerator.from_pretrained(checkpoint_dir, torch_dtype='auto', 
                                                        low_cpu_mem_usage=shared.model_args.low_cpu_mem)
    generator = generator.to(device=shared.device, non_blocking=True)
    generator.eval()
    return tokenizer, generator


def build_prompt(
    knowledge: str = '', 
    instruction: str = INSTRUCTION, 
    chat_history: list = [],
):
    if instruction.strip() != '':
        prompt = f"Instruction: {instruction}"
    else:
        prompt = ""

    if knowledge.strip() != '':
        prompt += f' [KNOWLEDGE] {knowledge}'

    if len(chat_history) > 0:
        chat_history = ' EOS '.join(chat_history)
        prompt += f" [CONTEXT] {chat_history}"
    return prompt


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


if __name__ == "__main__":

    from ..test import questionnaire

    gen_config = dict(max_length=8192, min_length=16, top_p=0.9, num_beams=2, do_sample=False)

    tokenizer, generator = load_model()

    dialog = []

    for role, question in questionnaire:
        prompt = build_prompt(
            # knowledge=role,
            instruction=INSTRUCTION, 
            chat_history=[question], 
        )
        response = generate_response(generator, tokenizer, prompt, **gen_config)

        print('\n\n')
        print(role)
        print('\t+', question)
        print('\t-', response)
        # print(prompt)
