from typing import Any, Callable
from .MMLU_data import MMLUBench
from .MMLU_program import CoT, RAG, SimplifiedBaleen
from langProBe.benchmark import Benchmark
import dspy

benchmark: Callable[[], Benchmark] = MMLUBench
programs = [CoT, RAG, SimplifiedBaleen]
metric = dspy.evaluate.answer_exact_match
