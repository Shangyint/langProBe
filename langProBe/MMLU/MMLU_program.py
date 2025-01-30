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
MMLUGeneratorCriticRanker_20 = program.GeneratorCriticRanker(GenerateAnswer, n=20)
MMLUGeneratorCriticFuser = program.GeneratorCriticFuser(GenerateAnswer)
MMLUGeneratorCriticFuser_20 = program.GeneratorCriticFuser(GenerateAnswer, n=20)
