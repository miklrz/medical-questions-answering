"""
A/B testing of prompts: ROUGE, BLEU, semantic similarity.
"""

import json
from pathlib import Path

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
import numpy as np


def load_test_set(path: str | Path = "data/test_questions.json") -> list[dict]:
    """Load test set of medical Q&A pairs."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def compute_rouge(reference: str, hypothesis: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def _ensure_nltk():
    try:
        import nltk

        nltk.download("punkt", quiet=True)
    except Exception:
        pass


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU score (sentence-level)."""
    _ensure_nltk()
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    smooth = SmoothingFunction()
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth.method1)


def compute_semantic_similarity(
    reference: str,
    hypothesis: str,
    model: SentenceTransformer | None = None,
) -> float:
    """Cosine similarity between embeddings."""
    if model is None:
        model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    ref_emb = model.encode([reference])
    hyp_emb = model.encode([hypothesis])
    return float(np.dot(ref_emb[0], hyp_emb[0]) / (np.linalg.norm(ref_emb[0]) * np.linalg.norm(hyp_emb[0]) + 1e-9))


def evaluate_response(
    reference: str,
    hypothesis: str,
    compute_semantic: bool = True,
) -> dict[str, float]:
    """Compute all metrics for a single Q&A pair."""
    metrics = {**compute_rouge(reference, hypothesis), "bleu": compute_bleu(reference, hypothesis)}
    if compute_semantic:
        metrics["semantic_similarity"] = compute_semantic_similarity(reference, hypothesis)
    return metrics


def run_ab_test(
    test_set: list[dict],
    prompt_variants: list[str],
    answer_fn: callable,
) -> dict[str, dict[str, float]]:
    """
    Run A/B test across prompt variants.
    answer_fn(question: str, variant: str) -> str
    Returns: {variant: {metric: mean_value}}
    """
    results: dict[str, list[dict[str, float]]] = {v: [] for v in prompt_variants}
    for item in test_set:
        question = item["question"]
        reference = item["reference_answer"]
        for variant in prompt_variants:
            hypothesis = answer_fn(question, variant)
            m = evaluate_response(reference, hypothesis)
            results[variant].append(m)

    aggregated: dict[str, dict[str, float]] = {}
    for variant, metrics_list in results.items():
        if not metrics_list:
            continue
        keys = metrics_list[0].keys()
        aggregated[variant] = {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}
    return aggregated
