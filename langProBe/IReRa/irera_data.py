from ..benchmark import Benchmark
import dspy
from .irera_utils import load_data


class IReRaBench(Benchmark):
    def init_dataset(self):
        # let the user download their data?
        return
        

    def create_splits(self):
        (self.train_examples,
        self.validation_examples,
        self.test_examples,
        _,
        _,
        _) = load_data()
