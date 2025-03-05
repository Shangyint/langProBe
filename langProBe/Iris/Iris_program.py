import dspy
import langProBe.dspy_program as dspy_program


class Sig(dspy.Signature):
    "Given the petal and sepal dimensions in cm, predict the iris species."

    petal_length = dspy.InputField()
    petal_width = dspy.InputField()
    sepal_length = dspy.InputField()
    sepal_width = dspy.InputField()
    answer = dspy.OutputField(desc="setosa, versicolor, or virginica")


IrisPredict = dspy_program.Predict(Sig)
IrisCot = dspy_program.CoT(Sig)
IrisGeneratorCriticRanker = dspy_program.GeneratorCriticRanker(Sig)
IrisGeneratorCriticRanker_20 = dspy_program.GeneratorCriticRanker(Sig, n=20)
IrisGeneratorCriticFuser = dspy_program.GeneratorCriticFuser(Sig)
IrisGeneratorCriticFuser_20 = dspy_program.GeneratorCriticFuser(Sig, n=20)
