from langProBe.gsm8k.gsm8k_utils import gsm8k_evaluate
from .gsm8k_data import GSM8KBench
from .gsm8k_program import CoT
from langProBe.benchmark import BenchmarkMeta
import dspy

benchmark = [BenchmarkMeta(GSM8KBench, [CoT], gsm8k_evaluate)]
