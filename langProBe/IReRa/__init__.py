from .irera_data import IReRaBench
from .irera_program import Infer, InferRetrieve, InferRetrieveRank
from .irera_utils import rp_at_k
from langProBe.benchmark import BenchmarkMeta


programs = [Infer, InferRetrieve, InferRetrieveRank]
benchmark = [BenchmarkMeta(IReRaBench, programs, rp_at_k)]
