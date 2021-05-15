from collections import defaultdict
from typing import Callable, Dict, Optional, List, Set

import numpy as np
import spacy
from mpire import WorkerPool
from quickumls import QuickUMLS

from umlspred.evaluation import Instance
from umlspred.constants import ALL_SEMTYPES


class BaselineQuickUmlsClassifier:
    def __init__(self, instance: QuickUMLS, n_workers: int) -> None:
        self.q = instance
        self.n_workers = n_workers

    @classmethod
    def load(
        cls,
        path_to_quickumls: str,
        accepted_semtypes: Optional[Set[str]] = None,
        threshold: float = 0.9,
        similarity_name: str = "jaccard",
        spacy_string: str = "en_core_sci_sm",
        best_match: bool = False,
        n_workers: int = 1,
    ) -> "QuickUMLSClassifier":
        if accepted_semtypes is None:
            accepted_semtypes = ALL_SEMTYPES
        q = QuickUMLS(
            path_to_quickumls, accepted_semtypes=accepted_semtypes, threshold=threshold, similarity_name=similarity_name
        )
        # Load the spacy model, Disable the NER and Parser.
        q.nlp = spacy.load(spacy_string, disable=("ner", "parser"))
        return cls(q, n_workers)

    @property
    def accepted_types(self) -> Set[str]:
        """Short for returning the accepted semantic types of a quickumls instance."""
        return set(self.q.accepted_semtypes)

    def _predict_single(self, text: str) -> List[Instance]:
        """
        Predict semantic types for a single text.

        :param text: The text to predict.
        :return: A list of tuples. The first item of each tuple is the character start and end position,
            the second the label.
        """
        matches = self.q.match(text, best_match=True)
        pred = []
        # For each match.
        for match in matches:
            semtype = list(match[0]["semtypes"])[0]
            pred.append(((match[0]["start"], match[0]["end"]), semtype))
        return pred

    def predict(self, texts: List[str], pbar: bool = True) -> List[List[Instance]]:
        """
        Predict the semtypes for each text.

        :param texts: The texts for which to predict.
        :return: A list of tuples containing start and end index and label.
        """
        with WorkerPool(n_jobs=self.n_workers) as pool:
            pred = pool.map(self._predict_single, texts, chunk_size=1, progress_bar=pbar)

        return pred


class QuickUMLSClassifier(BaselineQuickUmlsClassifier):

    FUNCS = {"max": np.max, "mean": np.mean, "sum": np.sum}

    def __init__(self, instance: QuickUMLS, pooling: str, priors: Optional[Dict[str, float]], n_workers: int) -> None:
        """
        :param instance: A valid quickUMLS installation.
        :param pooling: The name of the pooling function to use. Should be 'mean', 'max' or 'sum'.
        :param priors: None or a dictionary mapping from semantic types to class probabilities.
        :param n_workers: The number of workers to use during prediction.
        """
        super().__init__(instance, n_workers)
        if priors is not None:
            if set(priors.keys()) != self.accepted_types:
                raise ValueError("The set of priors != the set of accepted types.")
            if not np.isclose(sum(priors.values()), 1.0):
                raise ValueError("The priors do not sum to 1, and thus don't follow a probability distribution.")

        self.priors = priors
        if pooling not in self.FUNCS:
            raise ValueError(f"mode should be in {self.FUNCS}, is now {pooling}")
        self.pooling = pooling

    @property
    def pooling_function(self) -> Callable:
        """Get the appropriate function."""
        return self.FUNCS[self.pooling]

    @classmethod
    def load(
        cls,
        path_to_quickumls: str,
        accepted_semtypes: Optional[Set[str]] = None,
        threshold: float = 0.9,
        similarity_name: str = "jaccard",
        pooling: str = "mean",
        spacy_string: str = "en_core_sci_sm",
        priors: Optional[Dict[str, float]] = None,
        n_workers: int = 1,
    ) -> "QuickUMLSClassifier":
        """
        Load a QuickUMLSClassifier instance.

        :param path_to_quickumls: The path to a valid quickUMLS installation.
        :param accepted_semtypes: A set of accepted semantic types. If this is None, we revert to all semantic types.
        :param threshold: The threshold to accept.
        :param similarity_name: The name of the similarity function. Accepted are 'jaccard', 'overlap', 'cosine' and 'dice'.
        :param pooling: The name of the pooling function to use. Should be 'mean', 'max' or 'sum'.
        :param spacy_string: The string of the spacy model to use.
        :param priors: None or a dictionary mapping from semantic types to class probabilities.
        :param n_workers: The number of workers to use during prediction.
        :return: An initialized QuickUMLSClassifier.
        """
        # Fail early
        if pooling not in cls.FUNCS:
            raise ValueError(f"mode should be in {cls.FUNCS}, is now {pooling}")

        if accepted_semtypes is None:
            accepted_semtypes = ALL_SEMTYPES

        q = QuickUMLS(
            path_to_quickumls, accepted_semtypes=accepted_semtypes, threshold=threshold, similarity_name=similarity_name
        )
        # Load the spacy model, Disable the NER and Parser.
        q.nlp = spacy.load(spacy_string, disable=("ner", "parser"))
        return cls(q, pooling, priors, n_workers)

    def _predict_single(self, text: str) -> List[Instance]:
        """
        Predict semantic types for a single text.

        :param text: The text to predict.
        :return: A list of tuples. The first item of each tuple is the character start and end position,
            the second the label.
        """
        # Find matches using the QuickUMLS instance.
        matches = self.q.match(text, best_match=True)

        pred = []
        # For each match.
        for match in matches:

            # Keep track of scores per semtype.
            semtype = defaultdict(list)
            for m in match:
                # A single match can have multiple semtypes
                for st in filter(lambda x: x in self.accepted_types, m["semtypes"]):
                    semtype[st].append(m["similarity"])
            # Aggregate using the scoring function.
            semtype = {k: self.pooling_function(v) for k, v in semtype.items()}

            # If we have defined priors, apply them.
            if self.priors is not None:
                semtype = {k: v * self.priors[k] for k, v in semtype.items()}
            # Take the best semtype for this prediction.
            semtype = sorted(semtype.items(), key=lambda x: x[1], reverse=True)[0][0]
            # All matches have the same start and end position.
            pred.append(((match[0]["start"], match[0]["end"]), semtype))

        return pred
