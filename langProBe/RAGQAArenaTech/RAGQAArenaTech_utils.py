import dspy


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question based on the information we already have."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()
