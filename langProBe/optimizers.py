from dataclasses import dataclass
from typing import Any, Callable, Type
import dspy
import dspy.teleprompt
from functools import partial


@dataclass
class OptimizerConfig:
    optimizer: Type[dspy.teleprompt.Teleprompter]
    init_args: dict
    compile_args: dict
    langProBe_configs: dict

    def __str__(self):
        return f"""
[[
    Optimizer: {self.optimizer}
    init_args: {self.init_args}
    compile_args: {self.compile_args}
    langProBe_configs: {self.langProBe_configs}
]]
        """

    def __repr__(self):
        return self.__str__()


# Optimizer configuration formats:
DEFAULT_OPTIMIZERS = [
    OptimizerConfig(
        optimizer=dspy.teleprompt.BootstrapFewShot,
        init_args=dict(max_errors=5000, max_labeled_demos=2),
        compile_args=dict(),
        langProBe_configs=dict(use_valset=False, name="BootstrapFewShot"),
    ),
    OptimizerConfig(
        optimizer=dspy.teleprompt.BootstrapFewShotWithRandomSearch,
        init_args=dict(max_errors=5000, max_labeled_demos=2, num_threads=16),
        compile_args=dict(),
        langProBe_configs=dict(
            use_valset=True, name="BootstrapFewShotWithRandomSearch"
        ),
    ),
    OptimizerConfig(
        optimizer=dspy.teleprompt.MIPROv2,
        init_args=dict(max_errors=5000, num_threads=16),
        compile_args=dict(
            requires_permission_to_run=False,
            num_trials=25,
            max_bootstrapped_demos=4,
            max_labeled_demos=2,
        ),
        langProBe_configs=dict(use_valset=True, name="MIPROv2"),
    ),
]


def update_optimizer_from_list(
    optimizer_list: list[OptimizerConfig], optimizer: OptimizerConfig
) -> list[OptimizerConfig]:
    new_optimizer_list = []
    for optimizer_config in optimizer_list:
        if optimizer.optimizer == optimizer_config.optimizer:
            new_optimizer_list.append(optimizer)
        else:
            new_optimizer_list.append(optimizer_config)
    return new_optimizer_list


def create_optimizer(
    optimizer_config: OptimizerConfig, metric, num_threads=None
) -> tuple[Callable, dict]:
    optimizer = optimizer_config.optimizer
    init_args = optimizer_config.init_args
    if num_threads and "num_threads" in init_args:
        init_args["num_threads"] = num_threads
    compile_args = optimizer_config.compile_args
    langProBe_configs = optimizer_config.langProBe_configs
    optimizer = optimizer(metric=metric, **init_args)
    return partial(optimizer.compile, **compile_args), langProBe_configs
