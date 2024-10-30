from ..benchmark import Benchmark
from .scone_utils import load_scone
import random
import os


class SconeBench(Benchmark):
    def init_dataset(self):
        if not os.path.exists("langProBe/scone/ScoNe"):
            os.system(
                "git clone https://github.com/selenashe/ScoNe.git langProBe/scone/ScoNe"
            )

        all_train = load_scone("langProBe/scone/ScoNe/scone_nli/train")
        all_test = load_scone("langProBe/scone/ScoNe/scone_nli/test")

        self.dataset = all_train
        self.test_set = all_test
