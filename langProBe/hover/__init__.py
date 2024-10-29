from langProBe.benchmark import BenchmarkMeta
from .hover_data import hoverBench
from .hover_program import RetrieveMultiHop
from .hover_utils import discrete_retrieval_eval

benchmark = [BenchmarkMeta(hoverBench, [RetrieveMultiHop], discrete_retrieval_eval)]