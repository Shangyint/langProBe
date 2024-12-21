import os
import dspy
import json

from ..benchmark import Benchmark
from .AlfWorld_utils.alfworld_setup import ensure_alfworld_setup


class AlfWorldBench(Benchmark):
    def _find_files(self, path):
        # Recursively find all .tw-pddl files in the given path
        for root, _dirs, files in os.walk(str(path)):
            for file in files:
                if file.endswith(".tw-pddl"):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if 'solvable' in data and data['solvable']:
                        yield filepath

    def init_dataset(self):
        data_dir = ensure_alfworld_setup()

        train_filepaths = list(self._find_files(data_dir / "json_2.1.1/train"))
        valid_seen_filepaths = list(self._find_files(data_dir / "json_2.1.1/valid_seen"))
        valid_unseen_filepaths = list(self._find_files(data_dir / "json_2.1.1/valid_unseen"))
        valid_train_filepaths = list(self._find_files(data_dir / "json_2.1.1/valid_train"))

        assert len(train_filepaths) == 3553
        assert len(valid_seen_filepaths) == 140
        assert len(valid_unseen_filepaths) == 134

        self.train_set = [dspy.Example(game_file=filepath).with_inputs("game_file") for filepath in train_filepaths] # Use for training
        self.val_set = [dspy.Example(game_file=filepath).with_inputs("game_file") for filepath in valid_train_filepaths]  # Use for validation during bootstrapping
        self.dev_set = [dspy.Example(game_file=filepath).with_inputs("game_file") for filepath in valid_seen_filepaths]  # Use for Internal Testing
        self.test_set = [dspy.Example(game_file=filepath).with_inputs("game_file") for filepath in valid_unseen_filepaths]  # Use for final testing

        self.dataset = self.train_set + self.val_set + self.dev_set + self.test_set
