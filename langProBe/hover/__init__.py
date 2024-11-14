from langProBe.benchmark import BenchmarkMeta
from .hover_data import hoverBench
from .hover_program import RetrieveMultiHop, RetrieveMultiHop_Predict
from .hover_utils import discrete_retrieval_eval

benchmark = [BenchmarkMeta(hoverBench, [RetrieveMultiHop_Predict(), RetrieveMultiHop()], discrete_retrieval_eval)]