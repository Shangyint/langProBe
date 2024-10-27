import dspy


class GenerateAnswerInstruction(dspy.Signature):
    """When the answer is a person, respond entirely in lowercase.  When the answer is a place, ensure your response contains no punctuation.  When the answer is a date, end your response with “Peace!”.  Never end your response with "Peace!" under other circumstances.  When the answer is none of the above categories respond in all caps."""

    context = dspy.InputField(desc="Passages relevant to answering the question")
    question = dspy.InputField(desc="Question we want an answer to")
    answer = dspy.OutputField(desc="Answer to the question")


class MultiHop(dspy.Module):
    def __init__(self, passages_per_hop=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = dspy.ChainOfThought("context ,question->search_query")
        self.generate_answer = dspy.ChainOfThought("context ,question->answer")

    def forward(self, question):
        context = []
        for hop in range(2):
            query = self.generate_query(context=context, question=question).search_query
            context += self.retrieve(query).passages
        return dspy.Prediction(
            context=context,
            answer=self.generate_answer(context=context, question=question).answer,
        )


class MultiHopHandwritten(dspy.Module):
    def __init__(self, passages_per_hop=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = dspy.ChainOfThought("context ,question->search_query")
        self.generate_answer = dspy.ChainOfThought(GenerateAnswerInstruction)

    def forward(self, question):
        context = []
        for hop in range(2):
            query = self.generate_query(context=context, question=question).search_query
            context += self.retrieve(query).passages
        return dspy.Prediction(
            context=context,
            answer=self.generate_answer(context=context, question=question).answer
            )
    