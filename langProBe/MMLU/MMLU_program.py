import dspy
import langProBe.dspy_program as dspy_program
import langProBe.langchain_program as langchain_program

from .MMLU_data import input_kwargs, output_kwargs


class GenerateAnswer(dspy.Signature):
    """Answer multiple choice questions."""

    question = dspy.InputField()
    answer = dspy.OutputField(
        desc="Do not write explanations or additional text. Just select the answer."
    )


MMLUNaiveLangChain = langchain_program.NaiveLangChainProgram(
    input_kwargs, output_kwargs
)
MMLUPredict = dspy_program.Predict(GenerateAnswer)
MMLUCoT = dspy_program.CoT(GenerateAnswer)
MMLURAG = dspy_program.RAG(GenerateAnswer)
MMLUSimplifiedBaleen = dspy_program.SimplifiedBaleen(GenerateAnswer)
MMLUGeneratorCriticRanker = dspy_program.GeneratorCriticRanker(GenerateAnswer)
MMLUGeneratorCriticRanker_20 = dspy_program.GeneratorCriticRanker(GenerateAnswer, n=20)
MMLUGeneratorCriticFuser = dspy_program.GeneratorCriticFuser(GenerateAnswer)
MMLUGeneratorCriticFuser_20 = dspy_program.GeneratorCriticFuser(GenerateAnswer, n=20)
