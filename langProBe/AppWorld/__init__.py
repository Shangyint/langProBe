import os

from .AppWorld_data import AppWorldBench
from .AppWorld_utils.appworld_metric import appworld_metric
from .AppWorld_program import AppWorldReact

from langProBe.benchmark import BenchmarkMeta

from langProBe.optimizers import OptimizerConfig
import dspy
import dspy.teleprompt


api_key = os.environ["OPENAI_API_KEY"]

lm = dspy.LM("gpt-4o-mini", api_key=api_key)
embedder = dspy.Embedder(model="text-embedding-3-small", api_key=api_key)

dspy.configure(lm=lm, embedder=embedder)


knn_config = dict(
    max_errors=1000,
    max_labeled_demos=4,
    max_bootstrapped_demos=16,
    num_threads=8,
    embedder=embedder,
)


appworld_teacher = AppWorldReact(add_few_shot=True)
appworld_teacher._name = "AppWorldReactAugmented"
appworld_student = AppWorldReact(add_few_shot=False)
appworld_student._name = "AppWorldReact"

APPWORLD_OPTIMIZERS = [
    OptimizerConfig(
        optimizer=dspy.teleprompt.BootstrapKNNWithRandomSearch,
        init_args=knn_config,
        compile_args=dict(teacher=appworld_teacher),
        langProBe_configs=dict(use_valset=False, name="BootstrapKNNWithRandomSearch"),
    ),
    OptimizerConfig(
        optimizer=dspy.teleprompt.MIPROv2KNN,
        init_args=dict(**knn_config, auto="medium"),
        compile_args=dict(
            requires_permission_to_run=False,
            num_trials=20,
            max_bootstrapped_demos=2,
            max_labeled_demos=0,
            teacher=appworld_teacher,
        ),
        langProBe_configs=dict(use_valset=True, name="MIPROv2KNN"),
    ),
]


benchmark = [
    BenchmarkMeta(
        AppWorldBench,
        [appworld_student, appworld_teacher],
        appworld_metric,
        dataset_mode="full",
        optimizers=APPWORLD_OPTIMIZERS,
    )
]

os.environ["APPWORLD_ROOT"] = os.path.join(os.getcwd(), "langProBe", "AppWorld")
