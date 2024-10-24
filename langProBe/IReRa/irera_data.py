from ..benchmark import Benchmark
from .irera_utils import load_data
import subprocess


class IReRaBench(Benchmark):
    def init_dataset(self):
        result = subprocess.run(['bash', 'langProBe/IReRa/load_data.sh'], capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        (train_examples,
        validation_examples,
        test_examples,
        _,
        _,
        _) = load_data()
        self.dataset = train_examples + validation_examples + test_examples