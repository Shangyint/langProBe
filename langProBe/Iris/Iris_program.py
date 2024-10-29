import dspy
import langProBe.program as program


class Sig(dspy.Signature):
    "Given the petal and sepal dimensions in cm, predict the iris species."

    petal_length = dspy.InputField()
    petal_width = dspy.InputField()
    sepal_length = dspy.InputField()
    sepal_width = dspy.InputField()
    answer = dspy.OutputField(desc="setosa, versicolor, or virginica")


IrisCot = program.CoT(Sig)
IrisGeneratorCriticRanker = program.GeneratorCriticRanker(Sig)
IrisGeneratorCriticFuser = program.GeneratorCriticFuser(Sig)
