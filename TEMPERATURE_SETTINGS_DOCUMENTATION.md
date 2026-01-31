# Temperature Settings in DSPy Program for Diverse Generations

This document explains where and how temperature is configured in the DSPy program to ensure diverse generations in the LangProBe repository.

## Overview

Temperature is a parameter in language models that controls the randomness of predictions. Higher temperatures (closer to 1.0) produce more diverse outputs, while lower temperatures (closer to 0.0) produce more deterministic outputs. In LangProBe, temperature is strategically set in multiple locations to ensure diverse generations.

## Location 1: Rule Induction in Optimizers (`langProBe/optimizers.py`)

### Class: `RulesInductionProgramINFER`

**File:** `langProBe/optimizers.py`  
**Lines:** 160-173

### Purpose
This class is part of the `BootstrapFewShotInfer` optimizer and is responsible for inducing natural language rules from training examples.

### Implementation Details

```python
def forward(self, examples_text):
    original_temp = dspy.settings.lm.kwargs.get("temperature", 0.7)
    if self.teacher_settings:
        with dspy.settings.context(**self.teacher_settings):
            print("Using teacher settings")
            print(dspy.settings.lm.model)
            dspy.settings.lm.kwargs["temperature"] = random.uniform(0.9, 1.0)
            print(dspy.settings.lm.kwargs["temperature"])
            prediction = self.rules_induction(examples_text=examples_text)
    else:
        # print('Using default DSPy settings')
        # print(dspy.settings.lm.model)
        dspy.settings.lm.kwargs["temperature"] = random.uniform(0.9, 1.0)
        prediction = self.rules_induction(examples_text=examples_text)
        dspy.settings.lm.kwargs["temperature"] = original_temp
    natural_language_rules = prediction.natural_language_rules.strip()
    if self.verbose:
        print(natural_language_rules)
    return natural_language_rules
```

### Key Points

1. **Temperature Range:** `random.uniform(0.9, 1.0)`
   - This generates a random temperature between 0.9 and 1.0 for each rule induction
   - High temperature ensures diverse natural language rules

2. **Restoration:** The original temperature is saved and restored after generation (when not using teacher_settings)

3. **Why Diverse Rules Matter:**
   - The `BootstrapFewShotInfer` optimizer generates multiple candidate programs (default: 10)
   - Each candidate needs different rules to explore the optimization space
   - Diverse rules help find better instruction combinations

## Location 2: Heart Disease Classification (`langProBe/HeartDisease/HeartDisease_program.py`)

### Class: `HeartDiseaseClassify`

**File:** `langProBe/HeartDisease/HeartDisease_program.py`  
**Lines:** 48-51

### Purpose
This program creates an ensemble of classifiers with different temperatures to generate diverse opinions for heart disease prediction.

### Implementation Details

```python
def __init__(self):
    self.classify = [
        dspy.ChainOfThought(HeartDiseaseSignature, temperature=0.7 + i * 0.01)
        for i in range(3)
    ]
    self.vote = dspy.ChainOfThought(HeartDiseaseVote)
```

### Key Points

1. **Temperature Values:** `0.7, 0.71, 0.72`
   - Creates 3 classifiers with incrementally increasing temperatures
   - Each classifier acts as a "trainee doctor" with slightly different reasoning styles

2. **Ensemble Approach:**
   - Multiple classifiers generate diverse opinions
   - A voting mechanism aggregates these opinions
   - This mimics real-world medical consultation with multiple doctors

3. **Forward Method (lines 54-97):**
   - Collects opinions from all 3 classifiers
   - Formats them as trainee doctor opinions
   - Uses a voting predictor to make the final decision

## Summary

### Temperature Configuration Strategy

| Location | Temperature Value | Purpose |
|----------|------------------|---------|
| `RulesInductionProgramINFER` | `random.uniform(0.9, 1.0)` | Diverse rule generation for optimization |
| `HeartDiseaseClassify` | `0.7, 0.71, 0.72` | Ensemble of classifiers with slight variations |

### Why Diversity Matters

1. **Optimization:** In `RulesInductionProgramINFER`, diverse generations help explore different instruction formulations, increasing the chance of finding optimal prompts.

2. **Ensemble Methods:** In `HeartDiseaseClassify`, diverse opinions from multiple models with different temperatures create a more robust prediction system.

3. **Exploration vs Exploitation:** Higher temperatures encourage exploration of the solution space, which is crucial during:
   - Training/optimization phase (rule induction)
   - Ensemble prediction (multiple perspectives)

## Related Files

- `langProBe/optimizers.py` - Contains optimizer configurations and rule induction logic
- `langProBe/HeartDisease/HeartDisease_program.py` - Example of ensemble approach with temperature variation
- `langProBe/dspy_program.py` - Base classes for DSPy programs in LangProBe

## Notes

- The temperature modification in `RulesInductionProgramINFER` is temporary and restored after generation (in the non-teacher_settings case)
- The use of `random.uniform()` means each rule induction gets a slightly different temperature, adding another layer of diversity
- Other programs in the repository may use DSPy's default temperature settings unless explicitly overridden
