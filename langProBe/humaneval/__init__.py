import dspy.datasets
import dspy.datasets.gsm8k

from langProBe.benchmark import BenchmarkMeta
from .humaneval_data import HumanEvalBench

from .humaneval_program import *
from .humaneval_utils import human_eval_evaluate
import dspy

benchmark = [
    BenchmarkMeta(
        HumanEvalBench,
        [
            HumanEvalPredict,
            HumanEvalCoT,
            HumanEvalGeneratorCriticFuser,
            HumanEvalGeneratorCriticRanker,
            HumanEvalGeneratorCriticFuser_20,
            HumanEvalGeneratorCriticRanker_20,
        ],
        human_eval_evaluate,
        num_threads=1,
    )
]
