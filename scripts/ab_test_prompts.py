#!/usr/bin/env python3
"""
A/B тестирование промптов: базовый vs few-shot vs CoT.
Требует запущенный API (uvicorn) и LLM (LM Studio).
"""

import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.evaluation import load_test_set, run_ab_test, evaluate_response
from src.api.dataset import MedicalDataset
from src.api.retrieval import Retrieval
from src.api.models import SimilarityModel
from src.api.graph import build_medical_qa_graph

# Use smaller test if API unavailable
TEST_PATH = Path(__file__).resolve().parent.parent / "data" / "test_questions.json"


def get_answer_fn(retrieval, similarity_model):
    """Build answer function that uses graph with different prompt variants."""

    def answer_fn(question: str, variant: str) -> str:
        graph = build_medical_qa_graph(
            retrieval, similarity_model, system_variant=variant
        )
        result = graph.invoke({"user_question": question})
        ans = result.get("validated_answer")
        if ans:
            return ans.answer
        return ""

    return answer_fn


def main():
    os.environ.setdefault("HOST", "localhost")

    # Load dataset and build index
    dataset = MedicalDataset("ruslanmv/ai-medical-chatbot", split="train")
    retrieval = Retrieval()
    retrieval.build_index(
        dataset.get_qa_pairs(), cache_dir=os.getenv("EMBEDDINGS_CACHE_DIR")
    )
    similarity_model = SimilarityModel()

    test_set = load_test_set(TEST_PATH)
    if not test_set:
        print("Test set not found. Create data/test_questions.json")
        return

    answer_fn = get_answer_fn(retrieval, similarity_model)
    variants = ["base", "few_shot", "cot", "full"]

    print("Running A/B test... (this may take a while)")
    results = run_ab_test(test_set, variants, answer_fn)

    print("\n=== A/B Test Results ===\n")
    for variant, metrics in results.items():
        print(f"Variant: {variant}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print()

    # Best variant by ROUGE-L
    best = max(results.items(), key=lambda x: x[1].get("rougeL", 0))
    print(f"Best by ROUGE-L: {best[0]} (ROUGE-L={best[1]['rougeL']:.4f})")


if __name__ == "__main__":
    main()
