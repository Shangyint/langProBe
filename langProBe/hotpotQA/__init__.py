from typing import Any, Callable

import dspy.datasets
import dspy.evaluate
from .hotpot_data import HotpotQABench
from .hotpot_program import (
    HotPotQACoT,
    HotPotQARAG,
    HotPotQASimplifiedBaleen,
    HotPotQAGeneratorCriticRanker,
    HotPotQAGeneratorCriticFuser,
)
from langProBe.benchmark import BenchmarkMeta
import dspy


benchmark = [
    BenchmarkMeta(
        HotpotQABench,
        [
            HotPotQACoT,
            HotPotQARAG,
            HotPotQASimplifiedBaleen,
            HotPotQAGeneratorCriticRanker,
            HotPotQAGeneratorCriticFuser,
        ],
        dspy.evaluate.answer_exact_match,
    )
]
