import dspy
import langProBe.program as program


class LLMJudgeSignature(dspy.Signature):
    """
    Compare two responses to a question, and determine which is better.
    """

    question = dspy.InputField()
    response_A = dspy.InputField()
    response_B = dspy.InputField()

    answer = dspy.OutputField(desc="The better response, A>B or B>A")


JudgePredict = program.Predict(LLMJudgeSignature)
JudgeCoT = program.CoT(LLMJudgeSignature)
JudgeGeneratorCriticFuser = program.GeneratorCriticFuser(LLMJudgeSignature)
JudgeGeneratorCriticFuser_20 = program.GeneratorCriticFuser(LLMJudgeSignature, n=20)
JudgeGeneratorCriticRanker = program.GeneratorCriticRanker(LLMJudgeSignature)
JudgeGeneratorCriticRanker_20 = program.GeneratorCriticRanker(LLMJudgeSignature, n=20)

