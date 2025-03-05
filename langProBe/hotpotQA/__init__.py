from typing import Any, Callable

import dspy.datasets
import dspy.evaluate
from .hotpot_data import HotpotQABench
from .hotpot_program import *
from langProBe.benchmark import BenchmarkMeta
import dspy


benchmark = [
    BenchmarkMeta(
        HotpotQABench,
        [
            HotPotQAPredict,
            HotPotQACoT,
            HotPotQARAG,
            HotPotQASimplifiedBaleen,
            HotPotQAGeneratorCriticRanker,
            HotPotQAGeneratorCriticFuser,
            HotPotQAGeneratorCriticFuser_20,
            HotPotQAGeneratorCriticRanker_20,
        ],
        dspy.evaluate.answer_exact_match,
    )
]
