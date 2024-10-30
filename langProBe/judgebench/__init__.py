from .judgebench_data import JudgeBench
from .judgebench_program import JudgeCoT, JudgeGeneratorCriticFuser, JudgeGeneratorCriticRanker

from langProBe.benchmark import BenchmarkMeta

def llm_judge_eval(gold, pred, target: str = None):
    if ">" in pred.answer:
        pred_preference = pred.answer
    elif pred.answer.startswith('A'):
        pred_preference = "A>B"
    elif pred.answer.startswith('B'):
        pred_preference = "B>A"
    else:
        pred_preference = "A>B"

    return gold.label == pred_preference        

benchmark = [BenchmarkMeta(JudgeBench, [JudgeCoT, JudgeGeneratorCriticFuser, JudgeGeneratorCriticRanker], llm_judge_eval)]