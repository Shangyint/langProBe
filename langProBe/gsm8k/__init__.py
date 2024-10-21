from typing import Any, Callable

import dspy.datasets
import dspy.datasets.gsm8k
from .gsm8k_data import GSM8KBench
from .gsm8k_program import CoT
from langProBe.benchmark import Benchmark, BenchmarkMeta
import dspy

benchmark = [BenchmarkMeta(GSM8KBench, [CoT], dspy.datasets.gsm8k.gsm8k_metric)]
