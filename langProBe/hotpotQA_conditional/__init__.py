from .hotpot_conditional_data import HotpotQAConditionalBench
from .hotpot_conditional_program import (
    HotPotQACondSimplifiedBaleen,
    HotPotQACondSimplifiedBaleenHandwritten,
)
from .hotpot_conditional_utils import check_conditions
from langProBe.benchmark import BenchmarkMeta


benchmark = [
    BenchmarkMeta(
        HotpotQAConditionalBench,
        [HotPotQACondSimplifiedBaleen, HotPotQACondSimplifiedBaleenHandwritten],
        check_conditions,
    )
]
