import dspy
import langProBe.dspy_program as dspy_program


class LLMJudgeSignature(dspy.Signature):
    """
    Compare two responses to a question, and determine which is better.
    """

    question = dspy.InputField()
    response_A = dspy.InputField()
    response_B = dspy.InputField()

    answer = dspy.OutputField(desc="The better response, A>B or B>A")


JudgePredict = dspy_program.Predict(LLMJudgeSignature)
JudgeCoT = dspy_program.CoT(LLMJudgeSignature)
JudgeGeneratorCriticFuser = dspy_program.GeneratorCriticFuser(LLMJudgeSignature)
JudgeGeneratorCriticFuser_20 = dspy_program.GeneratorCriticFuser(
    LLMJudgeSignature, n=20
)
JudgeGeneratorCriticRanker = dspy_program.GeneratorCriticRanker(LLMJudgeSignature)
JudgeGeneratorCriticRanker_20 = dspy_program.GeneratorCriticRanker(
    LLMJudgeSignature, n=20
)
