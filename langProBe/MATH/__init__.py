from .MATH_utils import math_evaluate
from langProBe.benchmark import BenchmarkMeta
from .MATH_data import MATHBench
from .MATH_program import CoT


benchmark = [BenchmarkMeta(MATHBench, [CoT], math_evaluate)]
