from glob import glob
from pathlib import Path
from llama_cpp import Llama

from ... import shared
from ...path import MLM_LOCAL_MODELS


generation_config = {
    "max_tokens": 512,
    "temperature": 1.0,
    "repeat_penalty": 1.05,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.05,
}


def load_model(
    model_name: str,
    n_ctx=4096,         # Max sequence length to use. NOTE: longer sequence requires more resources
    n_threads=8,        # Number of CPU threads to use
    n_gpu_layers=-1,    # Number of layers to offload to GPU, if you have GPU acceleration available    
    verbose=True,
    **kwargs
):
    assert model_name in MLM_LOCAL_MODELS
    checkpoint_path = glob(MLM_LOCAL_MODELS[model_name]+'/*.gguf')[0]

    model = Llama(
        model_path=checkpoint_path, 
        n_ctx=n_ctx, 
        n_threads=n_threads, 
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
    return None, model


def build_prompt(chat_history: list = []):
    return dict(chat_history=chat_history)


def _build_prompt(
    content: str, 
    role: str = 'user', 
    chat_history: list = [],
):
    chat_history.append({"role": role, "content": content})
    return chat_history


def generate_response(
    generator,
    tokenizer = None,           # Keep it to match with Transformers library
    chat_history: list = [],
    **kwargs
):
    response = generator.create_chat_completion(chat_history, **kwargs)['choices'][0]['message']
    return response


def parse_response(response, role=None):
    if not role:
        return response
    return {'role': role, 'content': response['content']}


if __name__ == "__main__":

    from ..test import questionnaire, role_playing

    _, generator = load_model("llama-3.2-1b-instruct")
    # _, generator = load_model("natsumura-rp-chat")

    #############################
    #       Questionnaire       #
    #############################

    # for role, question in questionnaire:
    #     prompt = _build_prompt(role="system", content=role)
    #     prompt = _build_prompt(role="user", content=question, chat_history=prompt)
    #     response = generate_response(generator, chat_history=prompt, **generation_config)

    #     print('\n\n')
    #     print(role)
    #     print('\t+', question)
    #     print('\t-', response)

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
        response_1 = generate_response(generator, chat_history=player_1_memory, **generation_config)
        print("\n\n[Player 1]:", response_1['content'])
        player_1_memory.append(response_1)
        player_2_memory.append({"role": 'user', "content": response_1['content']})

        # Player-2 turn
        response_2 = generate_response(generator, chat_history=player_2_memory, **generation_config)
        print("\n\n[Player 2]:", response_2['content'])
        player_2_memory.append(response_2)
        player_1_memory.append({"role": 'user', "content": response_2['content']})


