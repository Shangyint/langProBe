from typing import Any, Callable
import dspy
import dspy.teleprompt
from functools import partial


# Optimizer configuration formats:
# (OptimizerClass, init_args, compile_args, langProBe_configs)
default_optimizers = [
    (
        dspy.teleprompt.BootstrapFewShot,
        dict(max_errors=1000, max_labeled_demos=2),
        dict(),
        dict(use_valset=False, name="BootstrapFewShot"),
    ),
    (
        dspy.teleprompt.BootstrapFewShotWithRandomSearch,
        dict(max_errors=1000, max_labeled_demos=2),
        dict(),
        dict(use_valset=True, name="BootstrapFewShotWithRandomSearch"),
    ),
    (
        dspy.teleprompt.MIPROv2,
        dict(max_errors=1000, auto="medium"),
        dict(
            requires_permission_to_run=False,
            num_trials=20,
            max_bootstrapped_demos=4,
            max_labeled_demos=2,
        ),
        dict(use_valset=True, name="MIPROv2"),
    ),
]


def create_optimizer(
    opt: Callable[[Any], dspy.teleprompt.Teleprompter],
    metric: callable,
    init_args,
    compile_args,
):
    optimizer = opt(metric=metric, **init_args)
    return partial(optimizer.compile, **compile_args)
