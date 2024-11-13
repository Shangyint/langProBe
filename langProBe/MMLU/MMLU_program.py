import dspy
import langProBe.program as program


class GenerateAnswer(dspy.Signature):
    """Answer multiple choice questions."""

    question = dspy.InputField()
    answer = dspy.OutputField(
        desc="Do not write explanations or additional text. Just select the answer."
    )


MMLUPredict = program.Predict(GenerateAnswer)
MMLUCoT = program.CoT(GenerateAnswer)
MMLURAG = program.RAG(GenerateAnswer)
MMLUSimplifiedBaleen = program.SimplifiedBaleen(GenerateAnswer)
MMLUGeneratorCriticRanker = program.GeneratorCriticRanker(GenerateAnswer)
MMLUGeneratorCriticFuser = program.GeneratorCriticFuser(GenerateAnswer)
