import dspy

import bm25s
import Stemmer
import ujson
import pickle
import os

corpus = []
with open("wiki.abstracts.2017.jsonl") as f:
    for line in f:
        line = ujson.loads(line)
        corpus.append(f"{line['title']} | {' '.join(line['text'])}")

stemmer = Stemmer.Stemmer("english")
tokenized_corpus_file = "corpus_tokens.pkl"
retriever_file = "bm25_index.pkl"

if os.path.exists(tokenized_corpus_file) and os.path.exists(retriever_file):
    with open(tokenized_corpus_file, "rb") as f:
        corpus_tokens = pickle.load(f)
    with open(retriever_file, "rb") as f:
        retriever = pickle.load(f)
else:
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25(k1=0.9, b=0.4)
    retriever.index(corpus_tokens)
    with open(tokenized_corpus_file, "wb") as f:
        pickle.dump(corpus_tokens, f)
    with open(retriever_file, "wb") as f:
        pickle.dump(retriever, f)

def search(query: str, k: int) -> dict:
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    run = {corpus[doc]: float(score) for doc, score in zip(results[0], scores[0])}
    return run


class HoverMultiHopPredict(dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.Predict("claim,summary_1->query")
        self.create_query_hop3 = dspy.Predict("claim,summary_1,summary_2->query")
        self.summarize1 = dspy.Predict("claim,passages->summary")
        self.summarize2 = dspy.Predict("claim,context,passages->summary")
        
    def forward(self, claim):
        # HOP 1
        hop1_docs_with_scores = search(claim, k=self.k)
        hop1_docs = list(hop1_docs_with_scores.keys())
        summary_1 = self.summarize1(claim=claim, passages=hop1_docs).summary

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs_with_scores = search(hop2_query, k=self.k)
        hop2_docs = list(hop2_docs_with_scores.keys())
        summary_2 = self.summarize2(claim=claim, context=summary_1, passages=hop2_docs).summary

        # HOP 3
        hop3_query = self.create_query_hop3(claim=claim, summary_1=summary_1, summary_2=summary_2).query
        hop3_docs_with_scores = search(hop3_query, k=self.k)
        hop3_docs = list(hop3_docs_with_scores.keys())

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


class HoverMultiHop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs_with_scores = search(claim, k=self.k)
        hop1_docs = list(hop1_docs_with_scores.keys())
        summary_1 = self.summarize1(claim=claim, passages=hop1_docs).summary

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs_with_scores = search(hop2_query, k=self.k)
        hop2_docs = list(hop2_docs_with_scores.keys())
        summary_2 = self.summarize2(claim=claim, context=summary_1, passages=hop2_docs).summary

        # HOP 3
        hop3_query = self.create_query_hop3(claim=claim, summary_1=summary_1, summary_2=summary_2).query
        hop3_docs_with_scores = search(hop3_query, k=self.k)
        hop3_docs = list(hop3_docs_with_scores.keys())

        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)
