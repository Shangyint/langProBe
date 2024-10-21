import dspy.datasets
import dspy.datasets.gsm8k

from langProBe.benchmark import BenchmarkMeta

from .swe_bench_verified_annotation_task_data import SWEBenchVerifiedAnnotationTaskBench

from .swe_bench_verified_underspecified_annotation_program import UnderspecifiedAnnotationGenerator
from .swe_bench_verified_underspecified_annotation_eval import underspecified_annotation_evaluate

from .swe_bench_verifier_eval_validity_annotation_program import EvaluationValidityModule
from .swe_bench_verifier_eval_validity_annotation_eval import evaluation_validity_evaluate

import dspy

benchmark = [
    BenchmarkMeta(SWEBenchVerifiedAnnotationTaskBench, [UnderspecifiedAnnotationGenerator], underspecified_annotation_evaluate),
    BenchmarkMeta(SWEBenchVerifiedAnnotationTaskBench, [EvaluationValidityModule], evaluation_validity_evaluate),
]
