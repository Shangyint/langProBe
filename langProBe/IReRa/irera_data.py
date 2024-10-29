from ..benchmark import Benchmark
from .irera_utils import load_data
import subprocess


class IReRaBench(Benchmark):
    def init_dataset(self):
        subprocess.run(
            ["bash", "langProBe/IReRa/load_data.sh"], capture_output=True, text=True
        )

        (train_examples, validation_examples, test_examples, _, _, _) = load_data()
        self.dataset = train_examples + validation_examples
        self.test_set = test_examples
