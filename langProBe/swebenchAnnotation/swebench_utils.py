import dspy


def underspecified_annotation_evaluate(
    example: dspy.Example, pred: dspy.Prediction, target: str = None
):
    score = 0
    if pred.underspecification_score in example.underspecified:
        score += 1

    return score


def evaluation_validity_evaluate(
    example: dspy.Example, pred: dspy.Prediction, target: str = None
):
    score = 0
    if pred.evaluation_validity_score in example.false_negative:
        score += 1

    return score
