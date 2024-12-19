import os

import dspy.teleprompt

from langProBe.benchmark import BenchmarkMeta
from langProBe.optimizers import OptimizerConfig

from .AlfWorld_data import AlfWorldBench
from .AlfWorld_program import (
    AlfWorldReAct,
    AlfWorldPredict,
    AlfWorldCoT,
)
from .AlfWorld_utils.alfworld_metric import alfworld_metric


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


ALFWORLD_OPTIMIZERS = [
    OptimizerConfig(
        optimizer=dspy.teleprompt.BootstrapKNN,
        init_args=knn_config,
        compile_args=dict(),
        langProBe_configs=dict(use_valset=False, name="BootstrapKNN"),
    ),
    OptimizerConfig(
        optimizer=dspy.teleprompt.BootstrapKNNWithRandomSearch,
        init_args=knn_config,
        compile_args=dict(),
        langProBe_configs=dict(use_valset=True, name="BootstrapKNNWithRandomSearch"),
    ),
    OptimizerConfig(
        optimizer=dspy.teleprompt.MIPROv2KNN,
        init_args=dict(**knn_config, auto="medium"),
        compile_args=dict(
            requires_permission_to_run=False,
        ),
        langProBe_configs=dict(use_valset=True, name="MIPROv2KNN"),
    ),
]


benchmark = [
    BenchmarkMeta(
        AlfWorldBench,
        [
            AlfWorldReAct(),
            # AlfWorldPredict(),
            # AlfWorldCoT()
        ],
        alfworld_metric,
        optimizers=ALFWORLD_OPTIMIZERS,
    ),
]
