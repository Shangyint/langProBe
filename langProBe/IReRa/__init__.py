from typing import Any, Callable
from .irera_data import IReRaBench
from .irera_program import Infer, InferRetrieve, InferRetrieveRank
from langProBe.benchmark import Benchmark, BenchmarkMeta
import dspy


programs = [Infer, InferRetrieve, InferRetrieveRank]

# TODO: should we change the k value
def rp_at_k(gold: list, predicted: list, k=10):
    """s
    Calculate Rank Precision at K (RP@K)

    Parameters:
    - gold: List containing the true relevant items
    - predicted: List containing the predicted items in ranked order
    - k: Top K items to consider

    Returns:
    - RP@K (Rank Precision at K) value
    """

    # Ensure k is not greater than the length of the gold list
    gold_k = min(k, len(gold))

    # Retrieve the top K predicted items
    top_k_predicted = predicted[:k]

    # Count the number of true positives in the top K
    true_positives = sum(1 for item in top_k_predicted if item in gold)

    # Calculate RP@K
    rp_at_k = true_positives / gold_k if gold_k > 0 else 0.0

    return rp_at_k

def p(gold, predicted):
    """Calculate the accuracy between two sets

    Parameters:
    - gold: The set of true labels
    - predicted: The set of predicted labels

    Returns:
    - Accuracy value
    """

    # import pdb
    # pdb.set_trace()
    predicted = predicted.predictions
    gold = set(gold.label)
    intersection = set(gold).intersection(set(predicted))

    accuracy = len(intersection) / len(gold) if len(gold) > 0 else 0.0

    return accuracy

benchmark = [
    BenchmarkMeta(
        IReRaBench, programs, p
    )
]
