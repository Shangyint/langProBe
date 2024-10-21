import dspy.datasets
import dspy.datasets.gsm8k
from langProBe.benchmark import BenchmarkMeta
from .MATH_data import MATHBench
from .MATH_program import CoT
import dspy


benchmark = [BenchmarkMeta(MATHBench, CoT, dspy.datasets.gsm8k.gsm8k_metric)]
