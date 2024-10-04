from abc import ABC, abstractmethod
import dspy
from typing import Any, Callable

from dspy.evaluate import Evaluate
from dspy.teleprompt import Teleprompter

from enum import Enum


class DSPyFeatures(Enum):
    BASELINE = 0
    OPTIMIZER = 1
    ASSERTION = 2


class Benchmark(ABC):
    def __init__(self):
        self.dataset = None
        self.init_dataset()
        self.create_splits()

    @abstractmethod
    def init_dataset(self) -> None:
        """
        Initializes the dataset for the benchmark, and sets it to self.dataset.
        """
        return

    @abstractmethod
    def create_splits(self) -> None:
        """
        Creates the splits for the dataset.
        Upon completion, self.train_set, self.dev_set, and self.test_set should be set.
        TODO(shangyin) shall we define a default split machnism?
        """
        return

    def get_dataset(self):
        return self.dataset

    def get_train_set(self):
        return self.train_set

    def get_dev_set(self):
        return self.dev_set

    def get_test_set(self):
        return self.test_set


class EvaluateBench(ABC):
    def __init__(
        self,
        benchmark: Benchmark,
        program: dspy.Module,
        metric: Callable,
        optimizers: list[Teleprompter] = None,
        has_assertions: bool = False,
        num_threads: int = 1,
    ):
        self.features: list[DSPyFeatures] = [DSPyFeatures.BASELINE]
        self.benchmark = benchmark
        self.program = program
        self.metric = metric
        self.optimizers = optimizers
        self.num_threads = num_threads
        self.evaluate_prog = Evaluate(
            devset=self.benchmark.get_dev_set(),
            metric=self.metric,
            num_threads=self.num_threads,
            display_progress=True,
        )

        if self.optimizer is not None:
            self.features.append(DSPyFeatures.OPTIMIZER)

        if has_assertions:
            self.features.append(DSPyFeatures.ASSERTION)

    def set_optimizer(self, optimizers: list[Teleprompter]) -> None:
        self.optimizers = optimizers

    def evaluate_baseline(self) -> float:
        return self.evaluate_prog(self.program)

    def evaluate_optimizers(self) -> list[float]:
        # TODO(shangyin): we need to pass additional arguments to the optimizer
        # one way is to create partial functions for optimizer in Teleprompter class, e.g.,
        # from functools import partial
        # def compile_partial(self, **kwargs):
        #     return partial(self.compile, **kwargs)
        #
        # and then we can pass optimizer.compile_partial as self.optimizer

        self.optimized_programs = [
            optimizer(self.program) for optimizer in self.optimizers
        ]

        return [
            self.evaluate_prog(optimized_program)
            for optimized_program in self.optimized_programs
        ]

    def evaluate_with_optimizer(self, optimizer: Teleprompter) -> float:
        optimized_program = optimizer(self.program)
        return self.evaluate_prog(optimized_program)

    def evaluate_assertion(self) -> float:
        self.program.activate_assertions()
        return self.evaluate_prog(self.program)

    def evaluate(self, dspy_config=None) -> dict[DSPyFeatures, float | list[float]]:
        """
        Args:
            dspy_config: A dictionary of configurations for dspy.context
        Returns:
            A dictionary of evaluation results for each feature.
            For baseline and assertion, the value is a float.
            For optimizer, the value is a list of floats corresponding to each optimizer.
        """
        if dspy_config is None:
            dspy_config = {}
        with dspy.context(**dspy_config):
            result: dict[DSPyFeatures, float] = {}
            for feature in self.features:
                match feature:
                    case DSPyFeatures.BASELINE:
                        result[feature] = self.evaluate_baseline()
                    case DSPyFeatures.OPTIMIZER:
                        result[feature] = self.evaluate_optimizers()
                    case DSPyFeatures.ASSERTION:
                        result[feature] = self.evaluate_assertion()
            return result
