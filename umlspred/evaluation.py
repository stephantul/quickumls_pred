from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

Pos = Tuple[int, int]
Instance = Tuple[Pos, str]
Score = Tuple[np.ndarray, np.ndarray, Tuple[str, np.ndarray]]


def to_iob(text: str, items: List[Instance]) -> List[str]:
    """Convert character level spans to IOB coding."""
    coding = ["O"] * len(text)
    for (s, e), label in items:
        b = f"B-{label}"
        i = f"I-{label}"
        coding[s] = b
        for x in range(s + 1, e):
            coding[x] = i

    return coding


def iob_conversion(text: str, gold: List[Instance], pred: List[Instance]) -> Tuple[List[str], List[str]]:
    """Does IOB conversion for gold and preds."""
    return to_iob(text, gold), to_iob(text, pred)


def in_interval(value: float, s: float, e: float) -> bool:
    """Check whether the value is within the interval defined by the predicate."""
    lower = value >= s
    upper = value <= e
    return lower and upper


def overlap(a: Pos, b: Pos, exact: bool = False) -> bool:
    """Calculate if two positions have overlap."""
    if a == b:
        return True
    elif exact:
        return False
    s0, e0 = a
    s1, e1 = b
    if in_interval(s1, s0, e0):
        return True
    if in_interval(e1, s0, e0):
        return True
    if in_interval(s0, s1, e1):
        return True
    if in_interval(e0, s1, e1):
        return True
    return False


def explain(gold: List[Instance], pred: List[Instance]) -> Dict[str, int]:
    """
    Explain predicted chunks.

    The idea behind this function is to get an overview of which kinds of mistakes a model makes.
    It outputs a dictionary with five categories and counts for these categories.
    """
    pred_score = np.zeros(len(pred))

    for idx, (pos, label) in enumerate(gold):
        for idx2, (pos2, label2) in enumerate(pred):
            a = overlap(pos, pos2, True)
            b = overlap(pos, pos2, False)
            if a or b:
                mod = b and not a
                if label == label2:
                    pred_score[idx2] = 1 + mod
                else:
                    pred_score[idx2] = 3 + mod

    out = {}
    out["false positive"] = sum(pred_score == 0)
    out["complete overlap, correct label"] = sum(pred_score == 1)
    out["overlap, correct label"] = sum(pred_score == 2)
    out["complete overlap, wrong label"] = sum(pred_score == 3)
    out["overlap, wrong label"] = sum(pred_score == 4)

    return out


def evaluate(gold: List[Instance], pred: List[Instance], exact: bool) -> Dict[str, Tuple[int, int, int]]:
    """
    Evaluate gold and predicted chunks for a single document.

    This function compares each gold chunk to each predicted chunk and counts them correct if their label is the same,
    and they have overlap.

    Overlap is defined depending on the exact flag. If this flag is False, we only need some overlap.
    If this is flag is True, we require the start and end indices to be equal.

    :param gold: The gold spans.
    :param pred: The predicted spans.
    :param exact: Whether to use exact or approximate matching.
    :return: A dictionary mapping from classes to true positives, false positives and false negatives.
    """
    # Initialize to zeros
    gold_score = np.zeros(len(gold))
    pred_score = np.zeros(len(pred))

    for idx, (pos, label) in enumerate(gold):
        for idx2, (pos2, label2) in enumerate(pred):
            if label == label2:
                if overlap(pos, pos2, exact):
                    gold_score[idx] = 1
                    pred_score[idx2] = 1
        # This ensures we only count one pred correctly for each gold.
        if gold_score[idx]:
            continue

    gold_labels = np.asarray([x[1] for x in gold])
    pred_labels = np.asarray([x[1] for x in pred])

    # Get the label set.
    label_set = set(gold_labels) | set(pred_labels)

    scores = {}
    for label in label_set:
        subset_gold = gold_score[gold_labels == label]
        subset_index = pred_score[pred_labels == label]
        tp = subset_gold.sum()
        fn = len(subset_gold) - tp
        fp = (subset_index == 0).sum()
        scores[label] = np.array([tp, fp, fn])

    return scores


def counts_to_scores(counts: Dict[str, Tuple[int, int, int]]) -> Score:
    """
    Converts a dictionary mapping from labels to true positives, false positives, and false negatives to scores

    :param counts: A dictionary mapping from a label string to a triple. The triple is true positives, false positives, false negatives.
    :return: macro-averaged PRF, micro-averaged PRF, and PRF per class.
    """
    labels, score = zip(*counts.items())
    score = np.stack(score)

    tp, fp, fn = score.T
    macro_p = tp / (fp + tp + 1e-16)
    macro_r = tp / (fn + tp + 1e-16)
    macro_f = 2 * macro_p * macro_r / (macro_p + macro_r + 1e-16)

    micro_p = tp.sum() / (fp.sum() + tp.sum() + 1e-16)
    micro_r = tp.sum() / (fn.sum() + tp.sum() + 1e-16)
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r)

    return (
        np.array([[macro_p.mean(), macro_r.mean(), macro_f.mean()], [micro_p, micro_r, micro_f]]),
        (labels, np.stack((macro_p, macro_r, macro_f))),
    )


def evaluate_all(
    data: List[Tuple[int, str, List[Instance]]], pred_data: List[List[Instance]],
) -> Tuple[Score, Score, Tuple[List[List[str]], List[List[str]]]]:
    """
    Evaluate model predictions on a set of documents using spans.

    This functions computes:
        - The inexact macro, micro and label, precision, recall and f-score.
        - The exact macro, micro, and label, prediction, recall, and f-score.
        - The character-based IOB predictions for gold and pred. These can be used to independently evaluate the model.

    :param data: The original data. Consists of triples of document ID, text, and List of gold labels.
    :param pred_data: The predictions. Consists of labels.
    :return: A tuple containing inexact scores, exact scores, and the gold and predicted IOB.
    """
    inexact_counts = defaultdict(lambda: np.zeros(3))
    exact_counts = defaultdict(lambda: np.zeros(3))

    iobs = []
    for (_, txt, gold), pred in zip(data, pred_data):

        for k, v in evaluate(gold, pred, exact=False).items():
            inexact_counts[k] += v
        for k, v in evaluate(gold, pred, exact=True).items():
            exact_counts[k] += v
        iobs.append(iob_conversion(txt, gold, pred))

    gold_iob, pred_iob = zip(*iobs)

    return (counts_to_scores(inexact_counts), counts_to_scores(exact_counts), (gold_iob, pred_iob))
