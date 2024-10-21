import dspy.datasets
import dspy.datasets.gsm8k

from langProBe.benchmark import BenchmarkMeta
from .humaneval_data import HumanEvalBench

from .humaneval_program import CoT
from .humaneval_utils import human_eval_evaluate
import dspy

benchmark = [BenchmarkMeta(HumanEvalBench, CoT, human_eval_evaluate)]
