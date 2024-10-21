from ..benchmark import Benchmark
import dspy
from .irera_utils import load_data


class IReRaBench(Benchmark):
    def init_dataset(self):
        # let the user download their data?
        (train_examples,
        validation_examples,
        test_examples,
        _,
        _,
        _) = load_data()
        self.dataset = train_examples + validation_examples + test_examples
        
