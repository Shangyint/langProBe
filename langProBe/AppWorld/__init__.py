from .AppWorld_data import AppWorldBench
from .AppWorld_utils.appworld_metric import appworld_metric
from .AppWorld_program import AppWorldReact

from langProBe.benchmark import BenchmarkMeta
import dspy

benchmark = [BenchmarkMeta(AppWorldBench, [AppWorldReact], appworld_metric)]
