import dspy


def alfworld_metric(example: dspy.Example, pred: dspy.Prediction, target: str = None):
    return pred.success
