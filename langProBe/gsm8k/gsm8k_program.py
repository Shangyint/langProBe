import dspy
import langProBe.dspy_program as dspy_program


class GenerateAnswerBasic(dspy.Signature):
    """
    Solve the problem step by step. List your reasoning for each step.
    """

    question = dspy.InputField()
    answer = dspy.OutputField(desc="The answer to the problem")


GSM8KPredict = dspy_program.Predict(GenerateAnswerBasic)
GSM8KCoT = dspy_program.CoT(GenerateAnswerBasic)
GSM8KGeneratorCriticFuser = dspy_program.GeneratorCriticFuser(GenerateAnswerBasic)
GSM8KGeneratorCriticRanker = dspy_program.GeneratorCriticRanker(GenerateAnswerBasic)

GSM8KGeneratorCriticFuser_20 = dspy_program.GeneratorCriticFuser(
    GenerateAnswerBasic, n=20
)
GSM8KGeneratorCriticRanker_20 = dspy_program.GeneratorCriticRanker(
    GenerateAnswerBasic, n=20
)
