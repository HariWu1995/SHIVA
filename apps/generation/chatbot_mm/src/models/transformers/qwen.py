import torch
from transformers import AutoTokenizer, AutoModelForCausalLM as AutoGenerator

from ... import shared
from ...path import MLM_LOCAL_MODELS


generation_config = dict(max_new_tokens=512)


def load_model(model_name: str):
    assert model_name in MLM_LOCAL_MODELS
    checkpoint_dir = MLM_LOCAL_MODELS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    generator = AutoGenerator.from_pretrained(checkpoint_dir, torch_dtype='auto', 
                                                        low_cpu_mem_usage=shared.model_args.low_cpu_mem)
    generator = generator.to(device=shared.device, non_blocking=True)
    generator.eval()
    return tokenizer, generator


build_prompt = None

def _build_prompt(
    content: str, 
    role: str = 'user', 
    chat_history: list = [],
):
    chat_history.append({"role": role, "content": content})
    return chat_history


def generate_response(
    generator,
    tokenizer,
    prompt: str | None = None,
    chat_history: list | None = None,
    **kwargs
):
    # Preprocessing
    if prompt is None:
        prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)

    with torch.no_grad():
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(device=shared.device)

        output = generator.generate(**inputs, **kwargs)[0]
        output = tokenizer.decode(output, skip_special_tokens=True)

    return output


def parse_response(output, role: str = 'assistant')
    return {'role': role, 'content': output}


if __name__ == "__main__":
    
    from ..test import questionnaire, role_playing

    tokenizer, generator = load_model("qwen-2.5-7b-instruct")

    #############################
    #       Questionnaire       #
    #############################

    for role, question in questionnaire:
        prompt = _build_prompt(role="system", content=role)
        prompt = _build_prompt(role="user", content=question, chat_history=prompt)
        response = generate_response(generator, tokenizer, chat_history=prompt, **generation_config)

        print('\n\n')
        print(role)
        print('\t+', question)
        print('\t-', response)

    #############################
    #        Role-playing       #
    #############################

    player_1_role = role_playing["player_1"]["role"]
    player_2_role = role_playing["player_2"]["role"]

    player_1_memory = _build_prompt(role="system", content=player_1_role)
    player_2_memory = _build_prompt(role="system", content=player_2_role)

    init_chat = role_playing["init_chat"]
    print("\n\n[Player 2]:", init_chat)

    player_1_memory = _build_prompt(role="user"     , content=init_chat, chat_history=player_1_memory)
    player_2_memory = _build_prompt(role="assistant", content=init_chat, chat_history=player_2_memory)

    for i in range(10):

        # Player-1 turn
        response_1 = generate_response(generator, tokenizer, chat_history=player_1_memory, **generation_config)
        print("\n\n[Player 1]:", response_1)
        player_1_memory.append({"role": 'assistant', "content": response_1})
        player_2_memory.append({"role":      'user', "content": response_1})

        # Player-2 turn
        response_2 = generate_response(generator, tokenizer, chat_history=player_2_memory, **generation_config)
        print("\n\n[Player 2]:", response_2)
        player_2_memory.append({"role": 'assistant', "content": response_2})
        player_1_memory.append({"role":      'user', "content": response_2})

