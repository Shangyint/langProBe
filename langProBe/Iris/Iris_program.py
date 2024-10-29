import dspy

class Sig(dspy.Signature):
    "Given the petal and sepal dimensions in cm, predict the iris species."

    petal_length = dspy.InputField()
    petal_width = dspy.InputField()
    sepal_length = dspy.InputField()
    sepal_width = dspy.InputField()
    answer = dspy.OutputField(desc="setosa, versicolor, or virginica")


class Classify(dspy.Module):
    def __init__(self):
        self.pred = dspy.ChainOfThought(Sig)

    def forward(self, petal_length, petal_width, sepal_length, sepal_width):
        return self.pred(
            petal_length=petal_length,
            petal_width=petal_width,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
        )
    

