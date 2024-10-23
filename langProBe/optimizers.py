from typing import Any, Callable
import dspy
import dspy.teleprompt
from functools import partial


def create_optimizer(
    opt: Callable[[Any], dspy.teleprompt.Teleprompter], metric: callable, init_args, compile_args
):
    optimizer = opt(metric=metric, **init_args)
    return partial(optimizer.compile, **compile_args)
