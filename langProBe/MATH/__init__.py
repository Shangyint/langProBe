from .MATH_utils import math_evaluate
from langProBe.benchmark import BenchmarkMeta
from .MATH_data import MATHBench
from .MATH_program import CoT, MultiChain, SelfCritic


benchmark = [BenchmarkMeta(MATHBench, [MultiChain, CoT, SelfCritic], math_evaluate)]
