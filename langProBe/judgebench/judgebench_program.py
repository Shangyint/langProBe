import dspy
from langProBe.program import CoT, GeneratorCriticFuser, GeneratorCriticRanker


class LLMJudgeSignature(dspy.Signature):
    """
    Compare two responses to a question, and determine which is better.
    """

    question = dspy.InputField()
    response_A = dspy.InputField()
    response_B = dspy.InputField()

    answer = dspy.OutputField(desc="The better response, A>B or B>A")

JudgeCoT = CoT(LLMJudgeSignature)
JudgeGeneratorCriticFuser = GeneratorCriticFuser(LLMJudgeSignature)
JudgeGeneratorCriticRanker = GeneratorCriticRanker(LLMJudgeSignature)
