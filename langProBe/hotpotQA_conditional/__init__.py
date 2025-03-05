from .hotpot_conditional_data import HotpotQAConditionalBench
from .hotpot_conditional_program import (
    HotPotQACondPredict,
    HotPotQACondSimplifiedBaleen,
    HotPotQACondSimplifiedBaleenHandwritten,
)
from .hotpot_conditional_utils import check_conditions
from langProBe.benchmark import BenchmarkMeta


benchmark = [
    BenchmarkMeta(
        HotpotQAConditionalBench,
        [
            HotPotQACondPredict,
            HotPotQACondSimplifiedBaleen,
            HotPotQACondSimplifiedBaleenHandwritten,
        ],
        check_conditions,
    )
]
