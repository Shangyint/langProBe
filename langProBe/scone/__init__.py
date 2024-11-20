from .scone_data import SconeBench
from .scone_program import (
    SconePredict,
    SconeCoT,
    SconeGeneratorCriticRanker,
    SconeGeneratorCriticFuser,
)
from langProBe.benchmark import BenchmarkMeta
import dspy


# benchmark = [
#     BenchmarkMeta(
#         SconeBench,
#         [SconePredict, SconeCoT, SconeGeneratorCriticRanker, SconeGeneratorCriticFuser],
#         dspy.evaluate.answer_exact_match,
#     )
# ]
benchmark = [
    BenchmarkMeta(
        SconeBench,
        [SconePredict],
        dspy.evaluate.answer_exact_match,
    )
]
