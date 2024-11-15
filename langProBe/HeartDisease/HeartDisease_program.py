import dspy
import langProBe.program as program

class HeartDiseaseInput(dspy.Signature):
    age = dspy.InputField(desc="Age in years")
    sex = dspy.InputField(desc="Sex (male or female)")
    cp = dspy.InputField(
        desc="Chest pain type (typical angina, atypical angina, non-anginal pain, asymptomatic)"
    )
    trestbps = dspy.InputField(
        desc="Resting blood pressure (in mm Hg on admission to the hospital)"
    )
    chol = dspy.InputField(desc="Serum cholestoral in mg/dl")
    fbs = dspy.InputField(desc="Fasting blood sugar > 120 mg/dl (true or false)")
    restecg = dspy.InputField(
        desc="Resting electrocardiographic results (normal, ST-T wave abnormality, left ventricular hypertrophy)"
    )
    thalach = dspy.InputField(desc="Maximum heart rate achieved")
    exang = dspy.InputField(desc="Exercise induced angina (yes or no)")
    oldpeak = dspy.InputField(desc="ST depression induced by exercise relative to rest")
    slope = dspy.InputField(
        desc="The slope of the peak exercise ST segment (upsloping, flat, downsloping)"
    )
    ca = dspy.InputField(desc="Number of major vessels (0-3) colored by flourosopy")
    thal = dspy.InputField(desc="Thalassemia (normal, fixed defect, reversible defect)")

class HeartDiseaseSignature(HeartDiseaseInput):
    """Given patient information, predict the presence of heart disease."""

    answer = dspy.OutputField(
        desc="Does this patient have heart disease? Just yes or no."
    )

class HeartDiseaseVote(HeartDiseaseInput):
    """Given patient information, predict the presence of heart disease. I can critically assess the provided trainee opinions."""

    context = dspy.InputField(desc="A list of opinions from trainee doctors.")
    answer = dspy.OutputField(
        desc="Does this patient have heart disease? Just yes or no."
    )


class Classify(dspy.Module):
    def __init__(self):
        self.classify = [
            dspy.ChainOfThought(HeartDiseaseSignature, temperature=0.7 + i * 0.01)
            for i in range(3)
        ]
        self.vote = dspy.ChainOfThought(HeartDiseaseVote)

    def forward(
        self,
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal,
    ):
        kwargs = dict(
            age=age,
            sex=sex,
            cp=cp,
            trestbps=trestbps,
            chol=chol,
            fbs=fbs,
            restecg=restecg,
            thalach=thalach,
            exang=exang,
            oldpeak=oldpeak,
            slope=slope,
            ca=ca,
            thal=thal,
        )


        opinions = [c(**kwargs) for c in self.classify]
        opinions = [
            (opinion.rationale.replace("\n", " ").strip("."), opinion.answer.strip("."))
            for opinion in opinions
        ]

        opinions = [
            f"I'm a trainee doctor, trying to {reason}. Hence, my answer is {answer}."
            for reason, answer in opinions
        ]
        
        return self.vote(context=opinions, **kwargs)
        

heartdiseasePredict = program.Predict(HeartDiseaseSignature)