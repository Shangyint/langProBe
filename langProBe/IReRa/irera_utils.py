import pandas as pd
import dspy
import random
import argparse
from typing import Union
import re
import os
from collections import Counter, defaultdict
import datasets
import json
import torch
import sentence_transformers
from sentence_transformers import SentenceTransformer
from functools import lru_cache

random.seed(1, version=2)

def normalize(
    label: str,
    do_lower: bool = True,
    strip_punct: bool = True,
    split_colon: bool = False,
) -> str:
    # Sometimes models wrongfully output a field-prefix, which we can remove.
    if split_colon:
        label = label.split(":")[1] if ":" in label else label

    # Remove leading and trailing newlines
    label = label.strip("\n")

    # Remove leading and trailing punctuation and newlines
    # NOTE: leading and trailing punctuation removal might hurt for e.g. drug and medical reaction ontologies.
    if strip_punct:
        label = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", label, flags=re.UNICODE)

    # Remove leading and trailing newlines
    label = label.strip("\n")

    # NOTE: lowering the labels might hurt for case-sensitive ontologies.
    if do_lower:
        return label.strip().lower()
    else:
        return label.strip()
    
def extract_labels_from_string(
    labels: str,
    do_lower: bool = True,
    strip_punct: bool = True,
    split_colon: bool = False,
) -> list[str]:
    return [
        normalize(r, do_lower=do_lower, strip_punct=strip_punct)
        for r in labels.split(",")
    ]

def extract_labels_from_strings(
    labels: list[str],
    do_lower: bool = True,
    strip_punct: bool = True,
    split_colon: bool = False,
) -> list[str]:
    labels = [
        normalize(
            r, do_lower=do_lower, strip_punct=strip_punct, split_colon=split_colon
        )
        for r in labels
    ]
    labels = ", ".join(labels)
    return extract_labels_from_string(
        labels, do_lower=do_lower, strip_punct=strip_punct, split_colon=split_colon
    )


def _prepare_esco_dataframe(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"sentence": "text"})

    # filter unusable rows
    df = df[df["label"] != "LABEL NOT PRESENT"]
    df = df[df["label"] != "UNDERSPECIFIED"]

    df["label"] = df["label"].apply(lambda x: [x])
    df = df.groupby("text").agg("sum").reset_index()
    df = df[["text", "label"]]

    df["label"] = df["label"].apply(lambda x: sorted(list(set(x))))

    return df


def _load_esco(task, validation_file, test_file):
    # base_dir = "./data"
    base_dir = "/Users/harry/Desktop/Nothing/sky/langProBe/langProBe/IReRa/data"
    esco_dir = os.path.join(base_dir, "esco")

    task_files = {
        "validation": os.path.join(esco_dir, validation_file)
        if validation_file
        else None,
        "test": os.path.join(esco_dir, test_file),
    }

    # get ontology
    # esco_skills, esco_descriptions, esco_priors = _load_esco_ontology(esco_dir)

    # get val and test set
    validation_df = (
        _prepare_esco_dataframe(task_files["validation"])
        if task_files["validation"]
        else None
    )
    test_df = _prepare_esco_dataframe(task_files["test"])

    return validation_df, test_df, None, None, None


def _prepare_biodex_dataframe(dataset):
    label = [
        extract_labels_from_string(
            l,
            do_lower=False,
            strip_punct=False,
        )
        for l in dataset["reactions"]
    ]
    df = pd.DataFrame({"text": dataset["fulltext_processed"], "label": label})
    return df


def _load_biodex():
    base_dir = "./data"
    biodex_dir = os.path.join(base_dir, "biodex")

    # get ontology
    biodex_terms = [
        term.strip("\n")
        for term in open(os.path.join(biodex_dir, "reaction_terms.txt")).readlines()
    ]

    # get val and test set
    dataset = datasets.load_dataset("BioDEX/BioDEX-Reactions")
    validation_ds, test_ds = dataset["validation"], dataset["test"]

    # get prior counts, normalized, from the train set
    all_train_reactions = dataset["train"]["reactions"]
    all_train_reactions = [ls.split(", ") for ls in all_train_reactions]
    all_train_reactions = [x for ls in all_train_reactions for x in ls]

    biodex_priors = Counter(all_train_reactions)
    biodex_priors = defaultdict(
        lambda: 0.0,
        {k: v / len(all_train_reactions) for k, v in biodex_priors.items()},
    )
    # save prior
    with open(os.path.join(biodex_dir, "biodex_priors.json"), "w") as fp:
        json.dump(biodex_priors, fp)

    # get correct format df[["text", "label"]]
    validation_df = _prepare_biodex_dataframe(validation_ds)
    test_df = _prepare_biodex_dataframe(test_ds)

    return validation_df, test_df, biodex_terms, None, biodex_priors


def get_dspy_examples(
    validation_df: Union[pd.DataFrame, None],
    test_df: pd.DataFrame,
    n_validation: int = None,
    n_test: int = None,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    valset, testset = [], []

    n_validation = float("+inf") if not n_validation else n_validation
    n_test = float("+inf") if not n_test else n_test

    if validation_df is not None:
        for _, example in validation_df.iterrows():
            if len(valset) >= n_validation:
                break
            valset.append(example.to_dict())
        valset = [dspy.Example(**x).with_inputs("text") for x in valset]
        # valsetX = [dspy.Example(**x).with_inputs('text', 'label') for x in valset]

    for _, example in test_df.iterrows():
        if len(testset) >= n_test:
            break
        testset.append(example.to_dict())
    testset = [dspy.Example(**x).with_inputs("text") for x in testset]
    # testsetX = [dspy.Example(**x).with_inputs('text', 'label') for x in testset]

    # print(len(valset), len(testset))
    return valset, testset


def load_data(dataset="esco_tech"):
    if dataset == "esco_house":
        (
            validation_df,
            test_df,
            ontology_items,
            ontology_descriptions,
            ontology_prior,
        ) = _load_esco(
            "house", "house_validation_annotations.csv", "house_test_annotations.csv"
        )
    elif dataset == "esco_tech":
        (
            validation_df,
            test_df,
            ontology_items,
            ontology_descriptions,
            ontology_prior,
        ) = _load_esco(
            "tech", "tech_validation_annotations.csv", "tech_test_annotations.csv"
        )
    elif dataset == "esco_techwolf":
        (
            validation_df,
            test_df,
            ontology_items,
            ontology_descriptions,
            ontology_prior,
        ) = _load_esco("techwolf", None, "techwolf_test_annotations.csv")
    elif dataset == "biodex_reactions":
        (
            validation_df,
            test_df,
            ontology_items,
            ontology_descriptions,
            ontology_prior,
        ) = _load_biodex()
    else:
        raise ValueError("Dataset not supported.")

    validation_examples, test_examples = get_dspy_examples(validation_df, test_df)

    # shuffle
    # NOTE: pull out this seed to get confidence intervals
    random.shuffle(validation_examples)
    random.shuffle(test_examples)

    # log some stats
    print(f"Dataset: {dataset}")

    print(f"# {dataset}: Total Validation size: {len(validation_examples)}")
    print(f"# {dataset}: Total Test size: {len(test_examples)}")
    # print(f"# {dataset}: Ontology items: {len(ontology_items)}")
    if "techwolf" not in dataset:
        print(
            f'{dataset}: avg # ontology items per input (for validation set): {round(validation_df["label"].apply(len).mean(),2)}'
        )
        print(
            f'{dataset}: Q25, Q50, Q75, Q95 # ontology items per input (for validation set): {validation_df["label"].apply(len).quantile([0.25, 0.5, 0.75, 0.95])}'
        )
    else:
        print(
            f'{dataset}: avg # ontology items per input (for test set): {round(test_df["label"].apply(len).mean(),2)}'
        )
        print(
            f'{dataset}: Q25, Q50, Q75, Q95 # ontology items per input (for test set): {test_df["label"].apply(len).quantile([0.25, 0.5, 0.75, 0.95])}'
        )

    # split off some of the validation examples for demonstrations
    # TODO: put these rangers in a config somewhere, or automate with random seed.
    if dataset == "esco_house" or dataset == "esco_tech":
        train_examples = validation_examples[:10]
        validation_examples = validation_examples[10:]
    elif dataset == "esco_techwolf":
        # techwolf has no validation data, use a mix of house and tech as proxy
        house_train, house_val, _, _, _, _ = load_data("esco_house")
        # tech_train, tech_val, _, _, _, _ = load_data("esco_tech")
        # train_examples = house_train + tech_train
        train_examples = house_train
        # validation_examples = house_val + tech_val
        validation_examples = house_val
        # shuffle train and val again
        random.shuffle(train_examples)
        random.shuffle(validation_examples)
    elif dataset == "biodex_reactions":
        # train_examples = validation_examples[:100]
        # validation_examples = validation_examples[100:200]
        # test_examples = test_examples[:500]
        train_examples = validation_examples[:10]
        validation_examples = validation_examples[100:150]
        test_examples = test_examples[:250]

    print(f"{dataset}: # Used Train size: {len(train_examples)}")
    print(f"{dataset}: # Used Validation size: {len(validation_examples)}")
    print(f"{dataset}: # Used Test size: {len(test_examples)}")

    return (
        train_examples,
        validation_examples,
        test_examples,
        ontology_items,
        ontology_descriptions,
        ontology_prior,
    )

class IreraConfig:
    """Every option in config should be serializable. No attribute should start with '_', since these are not saved."""

    def __init__(self, **kwargs):
        # signatures
        self.infer_signature_name = kwargs.pop("infer_signature_name")
        self.rank_signature_name = kwargs.pop("rank_signature_name")

        # hyperparameters
        self.prior_A = kwargs.pop("prior_A", 0)
        self.prior_path = kwargs.pop("prior_path", None)
        self.rank_topk = kwargs.pop("rank_topk", 50)
        self.chunk_context_window = kwargs.pop("chunk_context_window", 3000)
        self.chunk_max_windows = kwargs.pop("chunk_max_windows", 5)
        self.chunk_window_overlap = kwargs.pop("chunk_window_overlap", 0.02)

        # program logic flow
        self.rank_skip = kwargs.pop("rank_skip", False)

        # ontology
        self.ontology_path = kwargs.pop("ontology_path", None)
        self.ontology_name = kwargs.pop("ontology_name", None)
        self.retriever_model_name = kwargs.pop(
            "retriever_model_name", "sentence-transformers/all-mpnet-base-v2"
        )

        # optimizer
        self.optimizer_name = kwargs.pop("optimizer_name", None)

    def __repr__(self):
        return self.to_dict().__repr__()

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        return {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_") and not callable(value)
        }

    def to_json(self, filename):
        """Save the configuration to a JSON file."""
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    @classmethod
    def from_dict(cls, config_dict):
        """Create an instance of the configuration from a dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, filename):
        """Load the configuration from a JSON file."""
        with open(filename, "r") as file:
            config_dict = json.load(file)
        return cls.from_dict(config_dict)
    

class Retriever:
    def __init__(self, config: IreraConfig):
        self.config = config

        self.retriever_model_name = config.retriever_model_name
        self.friendly_model_name = self.retriever_model_name.replace("/", "--")

        self.ontology_name = config.ontology_name
        self.ontology_term_path = config.ontology_path

        # Initialize Retriever
        self.model = SentenceTransformer(self.retriever_model_name)
        self.model.to("cpu")

        # Initialize Ontology
        self.ontology_terms = self._load_terms()
        self.ontology_embeddings = self._load_embeddings()

    def _load_terms(self) -> list[str]:
        with open(self.ontology_term_path, "r") as fp:
            return [line.strip("\n") for line in fp.readlines()]

    def _load_embeddings(self) -> torch.Tensor:
        """Load or create embeddings for all query terms."""
        embedding_dir = os.path.join('.', 'data', 'embeddings')
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)
        ontology_embeddings_filename = os.path.join(embedding_dir,f"{self.ontology_name}_embeddings[{self.friendly_model_name}].pt")

        # If the file exists, load. Else, create embeddings.
        if os.path.isfile(ontology_embeddings_filename):
            with open(ontology_embeddings_filename, "rb") as f:
                ontology_embeddings = torch.load(f, map_location=torch.device("cpu"))
        else:
            self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            ontology_embeddings = self.model.encode(
                self.ontology_terms, convert_to_tensor=True, show_progress_bar=True
            )
            with open(ontology_embeddings_filename, "wb") as f:
                torch.save(ontology_embeddings, f)
            self.model.to(torch.device("cpu"))
        return ontology_embeddings

    @lru_cache(maxsize=100000)
    def retrieve_individual(self, query: str, K: int = 3) -> list[tuple[float, str]]:
        """Finds K closest matches based on semantic embedding similarity. Returns a list of (similarity_score, query) tuples."""
        query_embeddings = self.model.encode(query, convert_to_tensor=True)
        query_result = sentence_transformers.util.semantic_search(
            query_embeddings, self.ontology_embeddings, query_chunk_size=64, top_k=K
        )[0]

        # get (score, term) tuples
        matches = []
        for result in query_result:
            score = result["score"]
            term = self.ontology_terms[result["corpus_id"]]
            matches.append((score, term))

        return sorted(matches, reverse=True)

    def retrieve(self, queries: set[str]) -> dict[str, float]:
        """For every label in the ontology, get the maximum similarity over all queries. Returns a query --> max_score map."""

        queries = list(queries)

        # get similarities for each query
        query_embeddings = self.model.encode(queries, convert_to_tensor=True)
        query_results = sentence_transformers.util.semantic_search(
            query_embeddings,
            self.ontology_embeddings,
            query_chunk_size=64,
            top_k=len(self.ontology_embeddings),
        )

        # reformat results to be a query --> [score] map
        query_results_reformat = defaultdict(list)
        for query, query_result in zip(queries, query_results):
            for r in query_result:
                query = self.ontology_terms[r["corpus_id"]]
                query_score = r["score"]
                query_results_reformat[query].append(query_score)

        # for every query get the maximum score
        query_to_score = {k: max(v) for k, v in query_results_reformat.items()}

        return query_to_score


class Infer(dspy.Module):
    def __init__(self, config: IreraConfig):
        super().__init__()
        self.config = config
        self.cot = dspy.ChainOfThought(
            supported_signatures[config.infer_signature_name]
        )

    def forward(self, text: str) -> dspy.Prediction:
        parsed_outputs = set()

        output = self.cot(text=text).completions.output
        parsed_outputs.update(
            extract_labels_from_strings(output, do_lower=False, strip_punct=False)
        )

        return dspy.Prediction(predictions=parsed_outputs)
    


class Chunker:
    def __init__(self, config: IreraConfig):
        self.config = config
        self.chunk_context_window = config.chunk_context_window
        self.chunk_max_windows = config.chunk_max_windows
        self.chunk_window_overlap = config.chunk_window_overlap

    def __call__(self, text):
        snippet_idx = 0

        while snippet_idx < self.chunk_max_windows and text:
            endpos = int(self.chunk_context_window * (1.0 + self.chunk_window_overlap))
            snippet, text = text[:endpos], text[endpos:]

            next_newline_pos = snippet.rfind("\n")
            if (
                text
                and next_newline_pos != -1
                and next_newline_pos >= self.chunk_context_window // 2
            ):
                text = snippet[next_newline_pos + 1 :] + text
                snippet = snippet[:next_newline_pos]

            yield snippet_idx, snippet.strip()
            snippet_idx += 1



class Rank(dspy.Module):
    def __init__(self, config: IreraConfig):
        super().__init__()

        self.config = config
        self.cot = dspy.ChainOfThought(supported_signatures[config.rank_signature_name])

    def forward(self, text: str, options: list[str]) -> dspy.Predict:
        parsed_outputs = []

        output = self.cot(text=text, options=options).completions.output

        parsed_outputs = extract_labels_from_strings(
            output, do_lower=False, strip_punct=False, split_colon=True
        )

        return dspy.Prediction(predictions=parsed_outputs)
    




class InferSignatureESCO(dspy.Signature):
    __doc__ = f"""Given a snippet from a job vacancy, identify all the ESCO job skills mentioned. Always return skills."""

    text = dspy.InputField(prefix="Vacancy:")
    output = dspy.OutputField(
        prefix="Skills:",
        desc="list of comma-separated ESCO skills",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )


class RankSignatureESCO(dspy.Signature):
    __doc__ = f"""Given a snippet from a job vacancy, pick the 10 most applicable skills from the options that are directly expressed in the snippet."""

    text = dspy.InputField(prefix="Vacancy:")
    options = dspy.InputField(
        prefix="Options:",
        desc="List of comma-separated options to choose from",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )
    output = dspy.OutputField(
        prefix="Skills:",
        desc="list of comma-separated ESCO skills",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )


class InferSignatureBioDEX(dspy.Signature):
    __doc__ = f"""Given a snippet from a medical article, identify the adverse drug reactions affecting the patient. Always return reactions."""

    text = dspy.InputField(prefix="Article:")
    output = dspy.OutputField(
        prefix="Reactions:",
        desc="list of comma-separated adverse drug reactions",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )


class RankSignatureBioDEX(dspy.Signature):
    __doc__ = f"""Given a snippet from a medical article, pick the 10 most applicable adverse reactions from the options that are directly expressed in the snippet."""

    text = dspy.InputField(prefix="Article:")
    options = dspy.InputField(
        prefix="Options:",
        desc="List of comma-separated options to choose from",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )
    output = dspy.OutputField(
        prefix="Reactions:",
        desc="list of comma-separated adverse drug reactions",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )

supported_signatures = {
    "infer_esco": InferSignatureESCO,
    "rank_esco": RankSignatureESCO,
    "infer_biodex": InferSignatureBioDEX,
    "rank_biodex": RankSignatureBioDEX,
}


def rp_at_k(gold: list, predicted: list, k=50):
    """s
    Calculate Rank Precision at K (RP@K)

    Parameters:
    - gold: List containing the true relevant items
    - predicted: List containing the predicted items in ranked order
    - k: Top K items to consider

    Returns:
    - RP@K (Rank Precision at K) value
    """

    gold = gold.label
    predicted = list(predicted.predictions)

    # Ensure k is not greater than the length of the gold list
    gold_k = min(k, len(gold))

    # Retrieve the top K predicted items
    top_k_predicted = predicted[:k]

    # Count the number of true positives in the top K
    true_positives = sum(1 for item in top_k_predicted if item in gold)

    # Calculate RP@K
    rp_at_k = true_positives / gold_k if gold_k > 0 else 0.0

    return rp_at_k