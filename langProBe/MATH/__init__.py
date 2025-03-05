from .MATH_utils import math_evaluate, math_verify_evaluate
from langProBe.benchmark import BenchmarkMeta
from .MATH_data import MATHBench
from .MATH_program import *


benchmark = [
    BenchmarkMeta(
        MATHBench,
        [
            MATHPredict,
            MATHCoT,
            MATHGeneratorCriticFuser,
            MATHGeneratorCriticRanker,
        ],
        math_evaluate,
    ),
    BenchmarkMeta(
        MATHBench,
        [
            MATHPredict,
            MATHCoT,
            MATHGeneratorCriticFuser,
            MATHGeneratorCriticRanker,
        ],
        math_verify_evaluate,
        name="MATH-verify",
        num_threads=1,  # currently math-verify do not support being used multi-thread
    ),
]
