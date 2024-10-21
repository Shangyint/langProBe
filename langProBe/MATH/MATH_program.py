import dspy


class CoT(dspy.Module):
    def __init__(self):
        self.prog = dspy.ChainOfThought("problem -> answer")

    def forward(self, problem):
        pred = self.prog(problem=problem)
        return pred
