from .MATH_utils import math_evaluate
from langProBe.benchmark import BenchmarkMeta
from .MATH_data import MATHBench
from .MATH_program import (
    MATHPredict,
    MATHCoT,
    MATHGeneratorCriticFuser,
    MATHGeneratorCriticRanker,
)


benchmark = [
    BenchmarkMeta(
        MATHBench,
        [MATHPredict, MATHCoT, MATHGeneratorCriticFuser, MATHGeneratorCriticRanker],
        math_evaluate,
    )
]
