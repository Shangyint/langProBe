import dspy
import langProBe.dspy_program as dspy_program


class GenerateAnswerInstruction(dspy.Signature):
    """When the answer is a person, respond entirely in lowercase.  When the answer is a place, ensure your response contains no punctuation.  When the answer is a date, end your response with “Peace!”.  Never end your response with "Peace!" under other circumstances.  When the answer is none of the above categories respond in all caps."""

    question = dspy.InputField(desc="Question we want an answer to")
    answer = dspy.OutputField(desc="Answer to the question")


HotPotQACondPredict = dspy_program.Predict("question->answer")
HotPotQACondSimplifiedBaleen = dspy_program.SimplifiedBaleen("question->answer")
HotPotQACondSimplifiedBaleenHandwritten = dspy_program.SimplifiedBaleen(
    GenerateAnswerInstruction
)

# setting names for HotPotQACondSimplifiedBaleenHandwritten
HotPotQACondSimplifiedBaleenHandwritten._name = (
    "SimplifiedBaleenWithHandwrittenInstructions"
)
