from typing import Any, Callable
from gsm8k_data import GSM8KBench
from gsm8k_program import NaiveProgram
from langProBe.benchmark import Benchmark
import dspy

benchmark: Callable[[], Benchmark] = GSM8KBench

programs = [NaiveProgram]
