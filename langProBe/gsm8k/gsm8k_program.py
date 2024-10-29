import dspy
import langProBe.program as program


class GenerateAnswerBasic(dspy.Signature):
    """
    Solve the problem step by step. List your reasoning for each step.
    """

    question = dspy.InputField()
    answer = dspy.OutputField(desc="The answer to the problem")


GSM8KCoT = program.CoT(GenerateAnswerBasic)
GSM8KGeneratorCriticFuser = program.GeneratorCriticFuser(GenerateAnswerBasic)
GSM8KGeneratorCriticRanker = program.GeneratorCriticRanker(GenerateAnswerBasic)
