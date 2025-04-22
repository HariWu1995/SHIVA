from .. import shared
from ..path import MLM_LOCAL_MODELS, MULTIMODAL_ALLOWED


def prepare_tokenizer_and_generator(model_name: str, model_type: str = 'transformers'):

    assert model_name in MLM_LOCAL_MODELS
    assert model_type in ['transformers', 'llamacpp']

    shared.model_name = model_name
    shared.model_type = model_type

    # LlamaCpp
    if model_type == 'llamacpp':
        from .llamacpp import load_model, generate_response, build_prompt, generation_config, parse_response
    
    # Transformers
    elif 'deepseek' in model_name:
        if 'VL' in model_name:
            raise ImportError("Deepseek Vision-Language is not implemented yet!")
        from .transformers.deepseek import load_model, generate_response, build_prompt, generation_config, parse_response

    elif 'qwen' in model_name:
        if 'VL' in model_name:
            from .transformers.qwen_vl import load_model, generate_response, build_prompt, generation_config, parse_response
        else:
            from .transformers.qwen import load_model, generate_response, build_prompt, generation_config, parse_response

    elif 'moondream' in model_name:
        from .transformers.moondream import load_model, generate_response, build_prompt, generation_config, parse_response

    else:
        from .transformers.common import load_model, generate_response, build_prompt, generation_config, parse_response

    # Load generator and tokenizer / processor
    shared.generator, \
    shared.tokenizer = load_model(model_name) if model_type == 'transformers' \
                  else load_model(model_name, n_ctx = shared.model_args.max_seq_len, 
                                          n_threads = shared.model_args.n_threads, 
                                       n_gpu_layers = shared.model_args.n_gpu_layers, 
                                            verbose = shared.model_args.verbose)

    # Store generation variables
    shared.postprocess_fn    = parse_response
    shared.preparation_fn    = build_prompt
    shared.generation_fn     = generate_response
    shared.generation_config = generation_config
    
    # Multi-modal allowance
    shared.multimodals = MULTIMODAL_ALLOWED[model_name]

