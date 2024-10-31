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

        trainval_task_ids = load_task_ids("train") # 90
        dev_task_ids = load_task_ids("dev") # 57
        test_task_ids = load_task_ids("test_normal") # 168
        # test_task_ids += load_task_ids("test_challenge") # 417

        assert len(trainval_task_ids) == 90, len(trainval_task_ids)
        assert len(dev_task_ids) == 57
        # assert len(test_task_ids) == 168+417

        self.dev_set = [dspy.Example(task_id=task_id).with_inputs("task_id") for task_id in dev_task_ids]
        self.test_set = [dspy.Example(task_id=task_id).with_inputs("task_id") for task_id in test_task_ids]

        unique_scenario_ids_in_trainval = list(set([task_id.split('_')[0] for task_id in trainval_task_ids]))
        num_scenarios_in_val = int((len(unique_scenario_ids_in_trainval)+1)/2)

        rng = random.Random()
        rng.seed(1)
        train_scenarios = rng.sample(unique_scenario_ids_in_trainval, len(unique_scenario_ids_in_trainval)-num_scenarios_in_val)

        self.train_set = [dspy.Example(task_id=task_id).with_inputs("task_id") for task_id in trainval_task_ids if task_id.split('_')[0] in train_scenarios]
        self.val_set = [dspy.Example(task_id=task_id).with_inputs("task_id") for task_id in trainval_task_ids if task_id.split('_')[0] not in train_scenarios]
        assert len(self.val_set) >= len(self.train_set)
        
        self.dataset = self.train_set + self.dev_set + self.val_set
