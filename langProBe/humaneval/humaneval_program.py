import dspy

from itertools import chain
from .humaneval_utils import post_process_tests, post_process_code
import langProBe.dspy_program as dspy_program

NUM_SAMPLES = 20
TEMPARATURE_BASE = 0.7
TEMPARATURE_STEP = 0.01
NUM_TESTS = 5

# Code Generator


class CodeProblem(dspy.Signature):
    prompt = dspy.InputField(format=str)
    code = dspy.OutputField()


HumanEvalPredict = dspy_program.Predict(CodeProblem)
HumanEvalCoT = dspy_program.CoT(CodeProblem)
HumanEvalGeneratorCriticFuser = dspy_program.GeneratorCriticFuser(CodeProblem)
HumanEvalGeneratorCriticRanker = dspy_program.GeneratorCriticRanker(CodeProblem)

HumanEvalGeneratorCriticFuser_20 = dspy_program.GeneratorCriticFuser(CodeProblem, n=20)
HumanEvalGeneratorCriticRanker_20 = dspy_program.GeneratorCriticRanker(
    CodeProblem, n=20
)
