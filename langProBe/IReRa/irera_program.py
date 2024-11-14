import dspy
import math
import json
from .irera_utils import (
    Retriever,
    IreraConfig,
    Rank,
    Chunker,
    extract_labels_from_strings,
    supported_signatures,
)
import langProBe.program as program
import os


dir_path = os.path.dirname(os.path.abspath(__file__))
state_path = os.path.join(dir_path, "program_state.json")
state = json.load(open(state_path, "r"))
global_config = IreraConfig.from_dict(state["config"])


class ireraPredict(dspy.Module):
    def __init__(self, config: IreraConfig = global_config):
        super().__init__()
        self.config = config
        self.predict = program.Predict(supported_signatures[config.infer_signature_name])

    def forward(self, text):
        parsed_outputs = set()
        output = self.predict(text=text).completions.output
        parsed_outputs.update(
            extract_labels_from_strings(output, do_lower=False, strip_punct=False)
        )

        return dspy.Prediction(predictions=parsed_outputs)


class ireraCOT(dspy.Module):
    def __init__(self, config: IreraConfig = global_config):
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


class ireraRetrieve(dspy.Module):
    """Infer-Retrieve. Sets the Retriever, initializes the prior."""

    def __init__(self, config: IreraConfig = global_config):
        super().__init__()

        self.config = config

        # set LM predictor
        self.infer = ireraCOT(config)

        # set retriever
        self.retriever = Retriever(config)

        # set prior and prior strength
        self.prior = self._set_prior(config.prior_path)
        self.prior_A = config.prior_A

    def forward(self, text: str) -> dspy.Prediction:
        # Use the LM to predict label queries per chunk
        preds = self.infer(text).predictions

        # Execute the queries against the label index and get the maximal score per label
        scores = self.retriever.retrieve(preds)

        # Reweigh scores with prior statistics
        scores = self._update_scores_with_prior(scores)

        # Return the labels sorted
        labels = sorted(scores, key=lambda k: scores[k], reverse=True)

        return dspy.Prediction(
            predictions=labels,
        )

    def _set_prior(self, prior_path):
        """Loads the priors given a path and makes sure every term has a prior value (default value is 0)."""
        prior = json.load(open(prior_path, "r"))
        # Add 0 for every ontology term not in the file
        terms = self.retriever.ontology_terms
        terms_not_in_prior = set(terms).difference(set(prior.keys()))
        return prior | {t: 0.0 for t in terms_not_in_prior}

    def _update_scores_with_prior(self, scores: dict[str, float]) -> dict[str, float]:
        scores = {
            label: score * math.log(self.prior_A * self.prior[label] + math.e)
            for label, score in scores.items()
        }
        return scores


class ireraRetrieveRank(dspy.Module):
    """Infer-Retrieve-Rank, as defined in https://arxiv.org/abs/2401.12178."""

    def __init__(self, config: IreraConfig = global_config):
        super().__init__()

        self.config = config

        # Set Chunker
        self.chunker = Chunker(config)

        # Set InferRetrieve
        self.infer_retrieve = ireraRetrieve(config)

        # Set Rank
        self.rank = Rank(config)

        # Ranking hyperparameter
        self.rank_skip = config.rank_skip
        self.rank_topk = config.rank_topk

    def forward(self, text: str) -> dspy.Prediction:
        # Take the first chunk
        _, text = next(self.chunker(text))

        # Get ranking from InferRetrieve
        prediction = self.infer_retrieve(text)
        labels = prediction.predictions

        # Get candidates
        options = labels[: self.rank_topk]

        # Rerank
        if not self.rank_skip:
            predictions = self.rank(text, options).predictions

            # Only keep options that are valid
            selected_options = [o for o in predictions if o in options]

            # Supplement options
            selected_options = selected_options + [
                o for o in options if o not in selected_options
            ]
        else:
            selected_options = options

        return dspy.Prediction(
            predictions=selected_options,
        )

    def dump_state(self, verbose):
        """Dump the state. Uses the DSPy dump_state but also adds the config file."""
        return super().dump_state(verbose) | {"config": self.config.to_dict()}

    def load_state(self, state: dict):
        super().load_state(state)

    @classmethod
    def from_state(cls, state: dict):
        # get the config
        config = IreraConfig.from_dict(state["config"])
        # create a new program
        program = cls(config)
        # load the state
        program.load_state(state)
        return program

    @classmethod
    def load(cls, path: str):
        state = json.load(open(path, "r"))
        return cls.from_state(state)

