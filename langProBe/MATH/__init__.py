from .MATH_utils import math_evaluate
from langProBe.benchmark import BenchmarkMeta
from .MATH_data import MATHBench
from .MATH_program import *


benchmark = [
    BenchmarkMeta(
        MATHBench,
        [MATHPredict, MATHCoT, MATHGeneratorCriticFuser, MATHGeneratorCriticRanker, MATHGeneratorCriticFuser_20, MATHGeneratorCriticRanker_20],
        math_evaluate,
    )
]
