import dspy

from .irera_utils import extract_labels_from_strings

import math
import json

import dspy
from .irera_utils import Retriever

from .irera_utils import IreraConfig
from .irera_utils import Rank
from .irera_utils import Chunker

from .irera_utils import InferSignatureESCO, RankSignatureESCO, InferSignatureBioDEX, RankSignatureBioDEX, supported_signatures


class Infer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.config = None
        self.cot = dspy.ChainOfThought(InferSignatureESCO)

    def forward(self, text: str) -> dspy.Prediction:
        # import pdb
        # pdb.set_trace()
        parsed_outputs = set()

        output = self.cot(text=text).completions.output
        parsed_outputs.update(
            extract_labels_from_strings(output, do_lower=False, strip_punct=False)
        )

        return dspy.Prediction(predictions=parsed_outputs)
    

class InferRetrieve(dspy.Module):
    """Infer-Retrieve. Sets the Retriever, initializes the prior."""

    def __init__(
        self,
        config: IreraConfig,
    ):
        super().__init__()

        self.config = config

        # set LM predictor
        self.infer = Infer()

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
    


class InferRetrieveRank(dspy.Module):
    """Infer-Retrieve-Rank, as defined in https://arxiv.org/abs/2401.12178."""

    def __init__(
        self,
        config: IreraConfig,
    ):
        super().__init__()

        self.config = config

        # Set Chunker
        self.chunker = Chunker(config)

        # Set InferRetrieve
        self.infer_retrieve = InferRetrieve(config)

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

            # print(f"Rank returned {len(selected_options)} valid options.")

            # Supplement options
            selected_options = selected_options + [
                o for o in options if o not in selected_options
            ]
        else:
            selected_options = options

        return dspy.Prediction(
            predictions=selected_options,
        )

    def dump_state(self):
        """Dump the state. Uses the DSPy dump_state but also adds the config file."""
        return super().dump_state() | {"config": self.config.to_dict()}

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

    def save(self, path: str):
        state = self.dump_state()
        with open(path, "w") as fp:
            json.dump(state, fp)


