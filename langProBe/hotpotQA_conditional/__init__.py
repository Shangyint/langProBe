from .hotpot_conditional_data import HotpotQAConditionalBench
from .hotpot_conditional_program import MultiHop, MultiHopHandwritten
from .hotpot_conditional_utils import check_conditions
from langProBe.benchmark import BenchmarkMeta


benchmark = [
    BenchmarkMeta(
        HotpotQAConditionalBench, [MultiHop, MultiHopHandwritten], check_conditions
    )
]
