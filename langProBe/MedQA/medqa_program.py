import dspy
import langProBe.dspy_program as dspy_program

class MedQAAnswer(dspy.Signature):
    """You are an expert in answering medical exam questions."""
    
    question: str = dspy.InputField(description="The medical question to answer.")
    options: list = dspy.InputField(desc="List of multiple choice answer options to choose from.")
    answer: str = dspy.OutputField(description="The correct answer choice. Make sure to only provide the correct answer, and no additional text.")

MedQAPredict = dspy_program.Predict(MedQAAnswer)
MedQACoT = dspy_program.CoT(MedQAAnswer)