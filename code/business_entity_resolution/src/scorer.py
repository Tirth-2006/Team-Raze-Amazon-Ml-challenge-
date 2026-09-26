"""
Official F0.5 scorer - matches the competition's exact formula
"""
from typing import Dict, Set, Tuple


def _entity_f05(true_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F0.5 for a single entity

    Returns:
        (precision, recall, f05)
    """
    if not true_set:
        return (1.0, 1.0, 1.0) if not pred_set else (0.0, 0.0, 0.0)
    if not pred_set:
        return (0.0, 0.0, 0.0)

    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    if precision + recall == 0:
        return (precision, recall, 0.0)

    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return (precision, recall, f05)


def macro_f05(y_true: Dict[str, Set[str]], y_pred: Dict[str, Set[str]], return_details: bool = False):
    """
    Compute macro-averaged F0.5 score

    Args:
        y_true: Dict mapping S1 entity_id to set of matched S2/S3 ids
        y_pred: Dict mapping S1 entity_id to set of predicted S2/S3 ids
        return_details: If True, return (score, per_entity_details)

    Returns:
        macro_score (float) or (macro_score, per_entity_dict)
    """
    per_entity = {}
    for s1_id, true_set in y_true.items():
        pred_set = y_pred.get(s1_id, set())
        precision, recall, f05 = _entity_f05(set(true_set), set(pred_set))
        per_entity[s1_id] = {"precision": precision, "recall": recall, "f05": f05}

    scores = [v["f05"] for v in per_entity.values()]
    macro_score = sum(scores) / len(scores) if scores else 0.0

    return (macro_score, per_entity) if return_details else macro_score
