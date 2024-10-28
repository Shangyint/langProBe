from ..benchmark import Benchmark
from .AppWorld_utils.dataset_utils import load_task_ids
from .AppWorld_utils.appworld_setup import ensure_appworld_setup

import subprocess
import dspy
import sys
import os

class AppWorldBench(Benchmark):
    def init_dataset(self):
        ensure_appworld_setup()

        train_task_ids = load_task_ids("train")
        dev_task_ids = load_task_ids("dev")
        test_task_ids = load_task_ids("test_normal")
        test_task_ids += load_task_ids("test_challenge")

        self.train_set = [dspy.Example(task_id=task_id).with_inputs("task_id") for task_id in train_task_ids]
        self.dev_set = [dspy.Example(task_id=task_id).with_inputs("task_id") for task_id in dev_task_ids]
        self.test_set = [dspy.Example(task_id=task_id).with_inputs("task_id") for task_id in test_task_ids]
        self.dataset = self.train_set + self.dev_set + self.test_set

    def create_splits(self):
        # Set the number of examples to use for training, development, and testing to full dataset size
        # self.train_set = self.train_set[:5]
        # self.dev_set = self.dev_set[:5]
        # self.test_set = self.test_set[:5]
        pass
