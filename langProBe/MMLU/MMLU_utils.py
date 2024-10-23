import dspy


def deduplicate(seq: list[str]) -> list[str]:
    """
    Source: https://stackoverflow.com/a/480227/1493011
    """

    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def decide_model_type(model):
    if model.startswith("gpt"):
        return dspy.OpenAI(model=model, max_tokens=500)
    else:
        return dspy.HFClientVLLM(model=model, port=8080, max_tokens=500)

