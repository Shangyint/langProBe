import dspy
import langProBe.dspy_program as dspy_program

class MedMCQAAnswer(dspy.Signature):
    """You are an expert in answering medical exam questions."""
    
    question: str = dspy.InputField(description="The medical question to answer.")
    options: list = dspy.InputField(desc="List of multiple choice answer options to choose from.")
    answer: str = dspy.OutputField(description="The correct answer choice. Make sure to only provide the correct answer, and no additional text.")

MedMCQAPredict = dspy_program.Predict(MedMCQAAnswer)
MedMCQACoT = dspy_program.CoT(MedMCQAAnswer)