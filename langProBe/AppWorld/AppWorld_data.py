import random
import dspy

from ..benchmark import Benchmark
from .AppWorld_utils.dataset_utils import load_task_ids
from .AppWorld_utils.appworld_setup import ensure_appworld_setup


class AppWorldBench(Benchmark):
    def init_dataset(self):
        ensure_appworld_setup()

        trainval_task_ids = load_task_ids("train")
        dev_task_ids = load_task_ids("dev")
        dev_task_ids = [task_id for task_id in dev_task_ids]
        test_task_ids = load_task_ids("test_normal") # 168
        # test_task_ids += load_task_ids("test_challenge") # 417

        assert len(trainval_task_ids) == 90
        assert len(dev_task_ids) == 57
        assert len(test_task_ids) == 168

        # using simpler subset for training and validation
        self.dev_set = [dspy.Example(task_id=task_id).with_inputs("task_id") for task_id in dev_task_ids]
        self.test_set = [dspy.Example(task_id=task_id).with_inputs("task_id") for task_id in test_task_ids]

        unique_trainval_scenario_ids = list(set([task_id.split("_")[0] for task_id in trainval_task_ids]))
        rng = random.Random(0)
        rng.shuffle(unique_trainval_scenario_ids)
        num_train_scenarios = int(len(unique_trainval_scenario_ids) * 0.5)
        train_scenario_ids = unique_trainval_scenario_ids[:num_train_scenarios]
        val_scenario_ids = unique_trainval_scenario_ids[num_train_scenarios:]

        self.train_set = [dspy.Example(task_id=task_id).with_inputs("task_id") for task_id in trainval_task_ids if task_id.split("_")[0] in train_scenario_ids]
        self.val_set = [dspy.Example(task_id=task_id).with_inputs("task_id") for task_id in trainval_task_ids if task_id.split("_")[0] in val_scenario_ids]

        self.dataset = self.train_set + self.dev_set + self.val_set
