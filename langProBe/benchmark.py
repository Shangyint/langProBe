from abc import ABC, abstractmethod
import dspy
from typing import Callable

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
        optimizer: Teleprompter = None,
        has_assertions: bool = False,
        num_threads: int = 1,
    ):
        self.features: list[DSPyFeatures] = [DSPyFeatures.BASELINE]
        self.benchmark = benchmark
        self.program = program
        self.metric = metric
        self.optimizer = optimizer
        self.num_threads = num_threads
        self.evaluate_prog = Evaluate(
            devset=self.benchmark.get_dev_set(), metric=self.metric, num_threads=self.num_threads, display_progress=True
        )

        if self.optimizer is not None:
            self.features.append(DSPyFeatures.OPTIMIZER)

        if has_assertions:
            self.features.append(DSPyFeatures.ASSERTION)

    def evaluate_baseline(self) -> None:
        return self.evaluate_prog(self.program)

    def evaluate_optimizer(self) -> None:
        # TODO(shangyin): we need to pass additional arguments to the optimizer
        # one way is to create partial functions for optimizer in Teleprompter class, e.g.,
        # from functools import partial
        # def compile_partial(self, **kwargs):
        #     return partial(self.compile, **kwargs)
        #
        # and then we can pass optimizer.compile_partial as self.optimizer

        self.optimized_program = self.optimizer.compile(
            student=self.program, trainset=self.benchmark.get_train_set()
        )

        return self.evaluate_prog(self.optimized_program)

    def evaluate_assertion(self) -> None:
        self.program.activate_assertions()
        return self.evaluate_prog(self.program)

    def evaluate(self) -> None:
        result: dict[DSPyFeatures, float] = {}
        for feature in self.features:
            match feature:
                case DSPyFeatures.BASELINE:
                    result[feature] = self.evaluate_baseline()
                case DSPyFeatures.OPTIMIZER:
                    result[feature] = self.evaluate_optimizer()
                case DSPyFeatures.ASSERTION:
                    result[feature] = self.evaluate_assertion()
        return result
