from typing import Any, Callable
import dspy
import dspy.teleprompt
from functools import partial


def create_optimizer(
    opt: Callable[[Any], dspy.teleprompt.Teleprompter], metric: callable, **kwargs
):
    optimizer = opt(metric=metric)
    # increase optimizer's max errors
    optimizer.max_errors = kwargs.get("max_errors", 1000)
    return partial(optimizer.compile, **kwargs)
