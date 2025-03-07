import dspy
import langProBe.dspy_program as dspy_program

class PubMedQAAnswer(dspy.Signature):
    """You are an expert in answering medical research questions."""
    
    question: str = dspy.InputField(description="The medical question to answer.")
    answer: str = dspy.OutputField(description="The correct answer choice from yes/no/maybe. Make sure to only provide the correct answer, and no additional text.")

PubMedQAPredict = dspy_program.Predict(PubMedQAAnswer)
PubMedQACoT = dspy_program.CoT(PubMedQAAnswer)