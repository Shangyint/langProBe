import dspy

from langProBe.gsm8k.gsm8k_utils import gsm8k_evaluate
from langProBe.benchmark import BenchmarkMeta

from .gsm8k_data import GSM8KBench
from .gsm8k_program import (
    GSM8KPredict,
    GSM8KCoT,
    GSM8KGeneratorCriticFuser,
    GSM8KGeneratorCriticRanker,
)


benchmark = [
    BenchmarkMeta(
        GSM8KBench,
        [GSM8KPredict, GSM8KCoT, GSM8KGeneratorCriticFuser, GSM8KGeneratorCriticRanker],
        gsm8k_evaluate,
    )
]
