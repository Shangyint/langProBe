from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import random
import dspy
from typing import Any, Callable, List, Type

from dspy.evaluate import Evaluate
from dspy.teleprompt import Teleprompter

from enum import Enum

random.seed(1, version=2)

class DSPyFeatures(Enum):
    BASELINE = 0
    OPTIMIZER = 1
    ASSERTION = 2

dataset_size = {"full": None, "Lite": 500, "Tiny": 200}


class Benchmark(ABC):
    def __init__(self, dataset_mode="Lite"):
        # dataset for training and validation
        self.dataset = None
        # dataset for the actual benchmarking
        self.test_set = None
        self.train_set = None
        self.dev_set = None
        self.val_set = None

        self.init_dataset()
        assert self.dataset is not None, "Dataset not initialized"
        assert self.test_set is not None, "Test set not initialized"
        self.max_testset_size = dataset_size[dataset_mode]

        self.test_set = self.trim_dataset(self.test_set, self.max_testset_size)

        # TODO: FIXME: "Test" option is for debugging purposes only, should be removed for final release
        if dataset_mode == "Test":
            self.dataset = self.trim_dataset(self.dataset, 60)
            self.test_set = self.trim_dataset(self.test_set, 50)
        
        if not self.train_set or not self.dev_set or not self.val_set:
            self.create_splits()

        self.train_set = self.trim_dataset(self.train_set, 150)
        self.dev_set = self.trim_dataset(self.dev_set, 300)
        self.val_set = self.trim_dataset(self.val_set, 150)

        assert self.train_set is not None, "Train set not initialized"
        assert self.dev_set is not None, "Dev set not initialized"
        assert self.val_set is not None, "Val set not initialized"

    @abstractmethod
    def init_dataset(self) -> None:
        """
        Initializes the dataset for the benchmark, and sets it to self.dataset.
        Each element in the dataset should be an instance of dspy.Example.
        """
        return

    def trim_dataset(self, dataset, size: int) -> None:
        if size is None or size >= len(dataset):
            return dataset
        return random.sample(dataset, size)

    def create_splits(self) -> None:
        """
        Creates the splits for the dataset (not including test).
        Upon completion, self.train_set, self.dev_set, and self.val_set should be set.
        """

        total_len = len(self.dataset)
        self.dev_set = self.dataset[: int(0.5 * total_len)]
        self.val_set = self.dataset[int(0.5 * total_len) : int(0.75 * total_len)]
        self.train_set = self.dataset[int(0.75 * total_len) :]

    def get_dataset(self):
        return self.dataset

    def get_train_set(self):
        return self.train_set

    def get_dev_set(self):
        return self.dev_set

    def get_test_set(self):
        return self.test_set


@dataclass
class BenchmarkMeta:
    benchmark: Type[Benchmark]
    program: List[dspy.Module]
    metric: Callable
    dataset_mode: str = "Lite"


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
            devset=self.benchmark.get_test_set(),
            metric=self.metric,
            num_threads=self.num_threads,
            display_progress=True,
            # FIXME(shangyin): find a more ergonomic way to set max_errors
            max_errors=100,
        )

        self.results = None
        if self.optimizers is not None:
            self.features.append(DSPyFeatures.OPTIMIZER)

        if has_assertions:
            self.features.append(DSPyFeatures.ASSERTION)

    def set_optimizer(self, optimizers: list[Teleprompter]) -> None:
        self.optimizers = optimizers

    def evaluate_baseline(self) -> float:
        return self.evaluate_prog(self.program)

    def evaluate_optimizers(self) -> list[float]:
        self.optimized_programs = [
            optimizer(
                self.program,
                trainset=self.benchmark.train_set,
                valset=self.benchmark.val_set,
            )
            for optimizer in self.optimizers
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
            self.results = result
            return result
