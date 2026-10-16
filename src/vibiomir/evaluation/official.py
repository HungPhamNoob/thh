from __future__ import annotations

from collections import defaultdict
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


def lcs_overlap(candidate: str, gold: str) -> float:
    gold_tokens = official_normalize(gold).split()
    if not gold_tokens:
        return 0.0
    candidate_tokens = official_normalize(candidate).split()
    return _lcs_length(candidate_tokens, gold_tokens) / len(gold_tokens)


def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_squared = beta * beta
    return (1.0 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def _chunk_matches(predicted: list[dict], gold: list[dict], threshold: float) -> int:
    edges: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predicted):
        for gold_index, target in enumerate(gold):
            if str(pred["doc_id"]) != str(target["doc_id"]):
                continue
            overlap = lcs_overlap(str(pred["chunk_text"]), str(target["chunk_text"]))
            if overlap >= threshold:
                edges.append((overlap, pred_index, gold_index))
    matched_predictions: set[int] = set()
    matched_gold: set[int] = set()
    for _, pred_index, gold_index in sorted(edges, reverse=True):
        if pred_index not in matched_predictions and gold_index not in matched_gold:
            matched_predictions.add(pred_index)
            matched_gold.add(gold_index)
    return len(matched_gold)


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
    for gold in gold_rows:
        predicted = predicted_by_id.get(str(gold["id"]), {"relevant_docs": [], "relevant_chunks": []})
        gold_docs = {str(item) for item in gold.get("relevant_docs", [])}
        pred_docs = {str(item) for item in predicted.get("relevant_docs", [])}
        doc_p, doc_r = _pr(len(gold_docs & pred_docs), len(pred_docs), len(gold_docs))
        doc_precisions.append(doc_p)
        doc_recalls.append(doc_r)
        gold_chunks = gold.get("relevant_chunks", [])
        pred_chunks = predicted.get("relevant_chunks", [])
        matched = _chunk_matches(pred_chunks, gold_chunks, overlap_threshold)
        chunk_p, chunk_r = _pr(matched, len(pred_chunks), len(gold_chunks))
        chunk_precisions.append(chunk_p)
        chunk_recalls.append(chunk_r)
    macro = lambda values: sum(values) / len(values) if values else 0.0
    metrics = {
        "document_precision": macro(doc_precisions),
        "document_recall": macro(doc_recalls),
        "chunk_precision": macro(chunk_precisions),
        "chunk_recall": macro(chunk_recalls),
    }
    metrics["document_f2"] = f_beta(metrics["document_precision"], metrics["document_recall"])
    metrics["chunk_f2"] = f_beta(metrics["chunk_precision"], metrics["chunk_recall"])
    # Convenience aggregate for local sweeps. Keep official leaderboard composition configurable.
    metrics["retrieval_mean_f2"] = (metrics["document_f2"] + metrics["chunk_f2"]) / 2.0
    return metrics

