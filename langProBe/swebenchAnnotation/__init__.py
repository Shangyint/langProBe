from langProBe.benchmark import BenchmarkMeta
from .swebench_verified_annotation_task_data import SWEBenchVerifiedAnnotationTaskBench
from .swebench_verified_underspecified_annotation_program import (
    UnderspecifiedAnnotationCoT,
    UnderspecifiedAnnotationPredict,
)
from .swebench_verifier_eval_validity_annotation_program import (
    EvaluationValidityCoT,
    EvaluationValidityPredict,
)
from .swebench_utils import (
    evaluation_validity_evaluate,
    underspecified_annotation_evaluate,
)


benchmark = [
    BenchmarkMeta(
        SWEBenchVerifiedAnnotationTaskBench,
        [UnderspecifiedAnnotationPredict(), UnderspecifiedAnnotationCoT()],
        underspecified_annotation_evaluate,
    ),
    BenchmarkMeta(
        SWEBenchVerifiedAnnotationTaskBench,
        [EvaluationValidityPredict(), EvaluationValidityCoT()],
        evaluation_validity_evaluate,
    ),
]
