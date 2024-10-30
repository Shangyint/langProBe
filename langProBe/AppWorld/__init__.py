import os

from .AppWorld_data import AppWorldBench
from .AppWorld_utils.appworld_metric import appworld_metric
from .AppWorld_program import AppWorldReact

from langProBe.benchmark import BenchmarkMeta

# Keep the benchmark size full, since the full test set size is 585.
benchmark = [BenchmarkMeta(AppWorldBench, [AppWorldReact()], appworld_metric, dataset_mode="full")]

os.environ["APPWORLD_ROOT"] = os.path.join(os.getcwd(), "langProBe", "AppWorld")