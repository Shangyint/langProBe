import os
from .AlfWorld_utils.alfworld_setup import ensure_alfworld_setup
from ..benchmark import Benchmark
import dspy
import json

class AlfWorldBench(Benchmark):
    def find_solvable_pw_tddl_files(self, path):
        # Recursively find all .tw-pddl files in the given path
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".tw-pddl"):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if 'solvable' in data and data['solvable']:
                        yield filepath

    def init_dataset(self):
        ensure_alfworld_setup()

        train_filepaths = list(self.find_solvable_pw_tddl_files(os.path.join(os.environ["ALFWORLD_DATA"], 'json_2.1.1/train')))
        valid_seen_filepaths = list(self.find_solvable_pw_tddl_files(os.path.join(os.environ["ALFWORLD_DATA"], 'json_2.1.1/valid_seen')))
        valid_unseen_filepaths = list(self.find_solvable_pw_tddl_files(os.path.join(os.environ["ALFWORLD_DATA"], 'json_2.1.1/valid_unseen')))
        valid_train_filepaths = list(self.find_solvable_pw_tddl_files(os.path.join(os.environ["ALFWORLD_DATA"], 'json_2.1.1/valid_train')))

        assert len(train_filepaths) == 3553, f"Expected 3553 train files, got {len(train_filepaths)}"
        assert len(valid_seen_filepaths) == 140, f"Expected 140 valid seen files, got {len(valid_seen_filepaths)}"
        assert len(valid_unseen_filepaths) == 134, f"Expected 134 valid unseen files, got {len(valid_unseen_filepaths)}"

        self.train_set = [dspy.Example(game_file=filepath).with_inputs("game_file") for filepath in train_filepaths] # Use for training
        self.val_set = [dspy.Example(game_file=filepath).with_inputs("game_file") for filepath in valid_train_filepaths] # Use for validation during bootstrapping
        self.dev_set = [dspy.Example(game_file=filepath).with_inputs("game_file") for filepath in valid_seen_filepaths] # Use for Internal Testing
        self.test_set = [dspy.Example(game_file=filepath).with_inputs("game_file") for filepath in valid_unseen_filepaths] # Use for final testing

        self.dataset = self.train_set + self.val_set + self.dev_set + self.test_set
