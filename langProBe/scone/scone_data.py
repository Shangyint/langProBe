from ..benchmark import Benchmark
from .scone_utils import load_scone
import random
import os


class SconeBench(Benchmark):
    def init_dataset(self):
        if not os.path.exists("langProBe/scone/ScoNe"):
            os.system("git clone https://github.com/selenashe/ScoNe.git langProBe/scone/ScoNe")

        all_train = load_scone("langProBe/scone/ScoNe/scone_nli/train")

        random.seed(1)
        random.shuffle(all_train)

        self.dataset = all_train
