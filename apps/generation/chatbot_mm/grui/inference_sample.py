import time


def run_inference(state: dict, history: list):
    history.append({"role": "assistant", "content": ""})

    samples_text = [
        "\nWell.",
        "\nHmmm.",
        "\nLet me see ...",
        "\nYour are right!"
    ]
    samples_mm = [
        {"path": "C:/Users/Mr. RIAH/Documents/Projects/SHIVA/_samples/dollar-HV.jpg"},
        {"path": "C:/Users/Mr. RIAH/Documents/Projects/SHIVA/_samples/lip_sync.mp4"},
        {"path": "C:/Users/Mr. RIAH/Documents/Projects/SHIVA/_samples/Vũ Mập.mp3"},
    ]

    multi_responses = samples_text[:2] + samples_mm + samples_text[2:]
    for response in multi_responses:
        time.sleep(1)
        if isinstance(response, dict):
            history.append({"role": "assistant", "content": ""})
            history[-1]["content"] = response
        elif isinstance(history[-1]["content"], str):
            history[-1]["content"] += response
        else:
            history.append({"role": "assistant", "content": ""})
            history[-1]["content"] += response
        yield history


