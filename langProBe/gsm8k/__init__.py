from langProBe.gsm8k.gsm8k_utils import gsm8k_evaluate
from .gsm8k_data import GSM8KBench
from .gsm8k_program import *
from langProBe.benchmark import BenchmarkMeta
import dspy

benchmark = [
    BenchmarkMeta(
        GSM8KBench,
        [GSM8KPredict, GSM8KCoT, GSM8KGeneratorCriticFuser, GSM8KGeneratorCriticRanker, GSM8KGeneratorCriticRanker_20, GSM8KGeneratorCriticFuser_20],
        gsm8k_evaluate,
    )
]
