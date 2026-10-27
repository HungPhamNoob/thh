from __future__ import annotations

from typing import Any

from vibiomir.text import official_normalize


def _lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0]
        for index, left_token in enumerate(left, 1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def lcs_overlaps(candidate: str, gold: str) -> tuple[float, float]:
    """Return the two overlap ratios announced by the organizers on 2026-10-25.

    The first denominator is the gold chunk and controls recall.  The second
    denominator is the submitted chunk and prevents a whole document from being
    accepted merely because it contains the gold passage.
    """
    candidate_tokens = official_normalize(candidate).split()
    gold_tokens = official_normalize(gold).split()
    if not candidate_tokens or not gold_tokens:
        return 0.0, 0.0
    common = _lcs_length(candidate_tokens, gold_tokens)
    return common / len(gold_tokens), common / len(candidate_tokens)


def lcs_overlap(candidate: str, gold: str) -> float:
    """Backward-compatible name for the gold-normalized overlap."""
    return lcs_overlaps(candidate, gold)[0]


def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_squared = beta * beta
    return (1.0 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def _chunk_matches(predicted: list[dict], gold: list[dict], threshold: float) -> tuple[int, int]:
    """Return ``(correct_predictions, found_gold_chunks)`` without 1:1 matching.

    The competition counts every submitted chunk that matches at least one gold
    chunk for precision.  Recall independently counts every gold chunk matched
    by at least one submission.  Multiple predictions may therefore match the
    same gold chunk and vice versa.
    """
    matched_predictions: set[int] = set()
    matched_gold: set[int] = set()
    for pred_index, pred in enumerate(predicted):
        for gold_index, target in enumerate(gold):
            if str(pred["doc_id"]) != str(target["doc_id"]):
                continue
            overlap_gold, overlap_pred = lcs_overlaps(
                str(pred["chunk_text"]), str(target["chunk_text"])
            )
            if overlap_gold >= threshold and overlap_pred >= threshold:
                matched_predictions.add(pred_index)
                matched_gold.add(gold_index)
    return len(matched_predictions), len(matched_gold)


def _pr(tp: int, predicted: int, gold: int) -> tuple[float, float]:
    precision = tp / predicted if predicted else (1.0 if gold == 0 else 0.0)
    recall = tp / gold if gold else 1.0
    return precision, recall


def evaluate_submission(
    predictions: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    overlap_threshold: float = 0.4,
) -> dict[str, float]:
    predicted_by_id = {str(row["id"]): row for row in predictions}
    doc_precisions: list[float] = []
    doc_recalls: list[float] = []
    chunk_precisions: list[float] = []
    chunk_recalls: list[float] = []
    doc_f2s: list[float] = []
    chunk_f2s: list[float] = []
    for gold in gold_rows:
        predicted = predicted_by_id.get(
            str(gold["id"]), {"relevant_docs": [], "relevant_chunks": []}
        )
        gold_docs = {str(item) for item in gold.get("relevant_docs", [])}
        pred_docs = {str(item) for item in predicted.get("relevant_docs", [])}
        doc_p, doc_r = _pr(len(gold_docs & pred_docs), len(pred_docs), len(gold_docs))
        doc_precisions.append(doc_p)
        doc_recalls.append(doc_r)
        doc_f2s.append(f_beta(doc_p, doc_r))
        gold_chunks = gold.get("relevant_chunks", [])
        pred_chunks = predicted.get("relevant_chunks", [])
        correct_predictions, found_gold = _chunk_matches(
            pred_chunks, gold_chunks, overlap_threshold
        )
        chunk_p, chunk_r = _pr(correct_predictions, len(pred_chunks), len(gold_chunks))
        # Recall has a different numerator by definition.
        chunk_r = found_gold / len(gold_chunks) if gold_chunks else 1.0
        chunk_precisions.append(chunk_p)
        chunk_recalls.append(chunk_r)
        chunk_f2s.append(f_beta(chunk_p, chunk_r))

    def macro(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    metrics = {
        "document_precision": macro(doc_precisions),
        "document_recall": macro(doc_recalls),
        "chunk_precision": macro(chunk_precisions),
        "chunk_recall": macro(chunk_recalls),
    }
    # The leaderboard macro-averages per-query F2.  Computing F2 from the two
    # already-averaged P/R values gives a different (and incorrect) number.
    metrics["document_f2"] = macro(doc_f2s)
    metrics["chunk_f2"] = macro(chunk_f2s)
    metrics["final_score"] = (metrics["document_f2"] + metrics["chunk_f2"]) / 2.0
    metrics["retrieval_mean_f2"] = metrics["final_score"]
    return metrics
