import dspy.datasets
import dspy.datasets.gsm8k
from .humaneval_data import HumanEvalBench

from .humaneval_program import CoT
from .humaneval_utils import human_eval_evaluate
import dspy

benchmark = HumanEvalBench
programs = [CoT]
metric = human_eval_evaluate
