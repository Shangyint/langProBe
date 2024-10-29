import dspy

def count_unique_docs(example):
    return len(set([fact["key"] for fact in example["supporting_facts"]]))

def discrete_retrieval_eval(example, pred, trace=None):
    gold_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [doc["key"] for doc in example["supporting_facts"]],
        )
    )
    found_titles = set(
        map(
            dspy.evaluate.normalize_text,
            [c.split(" | ")[0] for c in pred.retrieved_docs],
        )
    )
    return gold_titles.issubset(found_titles)