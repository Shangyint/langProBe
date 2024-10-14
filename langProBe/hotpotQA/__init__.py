from typing import Any, Callable

import dspy.datasets
from dspy.datasets.hotpotqa import HotPotQA
from .hotpot_data import HotpotQABench
from .hotpot_program import CoT
from langProBe.benchmark import Benchmark
import dspy


benchmark: Callable[[], Benchmark] = HotpotQABench
programs = [CoT]
metric = dspy.evaluate.answer_exact_match
