from langProBe.benchmark import EvaluateBench
from langProBe.humaneval.humaneval_data import HumanEvalBench
from langProBe.humaneval.humaneval_program import NaiveCodeGenerator
from langProBe.humaneval.humaneval_utils import human_eval_evaluate
import dspy


human_eval_bench = HumanEvalBench()
evaluate_naive_program = EvaluateBench(
    human_eval_bench, NaiveCodeGenerator(), human_eval_evaluate
)

evaluate_naive_program.evaluate(
    dspy_config=dict(lm=dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=1000))
)
