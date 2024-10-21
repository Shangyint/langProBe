from typing import Any, Callable
import dspy
import dspy.teleprompt


def create_optimizer(
    opt: Callable[[Any], dspy.teleprompt.Teleprompter], metric: callable
):
    return opt(metric=metric)
