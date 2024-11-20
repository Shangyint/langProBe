from ..benchmark import Benchmark
from .AppWorld_utils.dataset_utils import load_task_ids
from .AppWorld_utils.appworld_setup import ensure_appworld_setup

import subprocess
import random
import dspy
import sys
import os

class AppWorldBench(Benchmark):
    def init_dataset(self):
        ensure_appworld_setup()

        trainval_task_ids = load_task_ids("train", difficulty=1)
        dev_task_ids = load_task_ids("dev", difficulty=1)
        test_task_ids = load_task_ids("test_normal") # 168
        # test_task_ids += load_task_ids("test_challenge") # 417

        # using simpler subset for training and validation
        self.dev_set = [dspy.Example(task_id=task_id).with_inputs("task_id") for task_id in dev_task_ids]
        self.test_set = [dspy.Example(task_id=task_id).with_inputs("task_id") for task_id in test_task_ids]
        self.train_set = [dspy.Example(task_id=task_id).with_inputs("task_id") for task_id in trainval_task_ids]
        self.val_set = self.dev_set

        self.dataset = self.train_set + self.dev_set + self.val_set
