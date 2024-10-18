from typing import Any, Callable
import dspy
import dspy.teleprompt


def create_optimizer(
    opt: Callable[[Any], dspy.teleprompt.Teleprompter], metric: callable
):
    print(type(opt))
    return opt(metric=metric)
