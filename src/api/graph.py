"""
LangGraph pipeline: Query Analysis → Retrieval → Reranking → Answer Generation → Quality Check.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END
import os

from src.api.schemas import MedicalAnswer
from src.api.prompts import build_generation_prompt
from src.api.retrieval import Retrieval
from src.api.models import SimilarityModel
from src.api.llm_generation import generate_structured
import logging


class GraphState(TypedDict):
    """State for the Medical QA graph."""

    user_question: str
    analyzed_query: str
    retrieved_docs: list[dict]
    reranked_docs: list[dict]
    contexts: str
    raw_answer: str | None
    validated_answer: MedicalAnswer | None
    confidence: float
    need_clarification: bool


def query_analysis_node(state: GraphState) -> dict:
    """Analyze query: extract intent, detect urgency."""
    logging.info("Start query analysis")
    query = state["user_question"].strip()
    urgency_keywords = [
        "боль",
        "кровь",
        "температура",
        "срочно",
        "скорая",
        "сердце",
        "голова",
        "передозировка",
        "потеря сознания",
        "не дышит",
    ]
    has_urgency = any(kw in query.lower() for kw in urgency_keywords)
    return {
        "analyzed_query": query,
        "need_clarification": False,
        "has_urgency": has_urgency,  # Fix: was computed but never returned
    }


def retrieval_node(state: GraphState, retrieval: Retrieval) -> dict:
    logging.info("Retrieving top-k candidates from FAISS")
    """Retrieve top-k candidates from FAISS."""
    query = state["analyzed_query"]
    candidates = retrieval.query(query, top_k=5)
    return {"retrieved_docs": candidates}


def reranking_node(state: GraphState, similarity_model: SimilarityModel) -> dict:
    """Rerank with BERT, keep top-3."""
    logging.info("Reranking")
    query = state["analyzed_query"]
    docs = state["retrieved_docs"]
    scored = []
    for d in docs:
        score = similarity_model.predict(query, d["text"])
        scored.append({"text": d["text"], "sim": score})
    top_n = sorted(scored, key=lambda x: x["sim"], reverse=True)[:3]
    contexts = "\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(top_n)])
    return {"reranked_docs": top_n, "contexts": contexts}


def answer_generation_node(
    state: GraphState,
    retrieval: Retrieval,
    similarity_model: SimilarityModel,
    system_variant: str = "full",
) -> dict:
    logging.info("Generating answer")
    """Generate answer via LLM with structured output."""
    contexts = state["contexts"]
    question = state["user_question"]
    prompt = build_generation_prompt(contexts, question, system_variant)
    sources = [c["text"] for c in state["reranked_docs"]]

    result = generate_structured(prompt, sources)
    logging.info("Answer generated")
    return {
        "validated_answer": result,
        "confidence": result.confidence,
    }


def quality_check_node(state: GraphState) -> dict:
    """Check quality: low confidence or urgency → force doctor visit."""
    confidence = state.get("confidence", 0.0)
    has_urgency = state.get("has_urgency", False)
    answer = state.get("validated_answer")

    need_clarification = confidence < 0.7 and answer is not None

    if answer:
        if need_clarification:
            answer.warnings = list(answer.warnings or []) + [
                "Уверенность модели в ответе низкая. Рекомендуется уточнить вопрос или обратиться к специалисту."
            ]
            answer.requires_doctor_visit = True

        if has_urgency:
            answer.requires_doctor_visit = True
            if (
                "Обнаружены признаки срочного состояния. Обратитесь за медицинской помощью."
                not in (answer.warnings or [])
            ):
                answer.warnings = list(answer.warnings or []) + [
                    "Обнаружены признаки срочного состояния. Обратитесь за медицинской помощью немедленно."
                ]

    return {"need_clarification": need_clarification, "validated_answer": answer}


def build_medical_qa_graph(
    retrieval: Retrieval,
    similarity_model: SimilarityModel,
    system_variant: str = "full",
) -> StateGraph:
    """Build LangGraph for Medical QA."""

    def _retrieval(s: GraphState) -> dict:
        return retrieval_node(s, retrieval)

    def _reranking(s: GraphState) -> dict:
        return reranking_node(s, similarity_model)

    def _generation(s: GraphState) -> dict:
        return answer_generation_node(s, retrieval, similarity_model, system_variant)

    workflow = StateGraph(GraphState)

    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieval", _retrieval)
    workflow.add_node("reranking", _reranking)
    workflow.add_node("answer_generation", _generation)
    workflow.add_node("quality_check", quality_check_node)

    workflow.set_entry_point("query_analysis")
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "reranking")
    workflow.add_edge("reranking", "answer_generation")
    workflow.add_edge("answer_generation", "quality_check")
    workflow.add_edge("quality_check", END)

    return workflow.compile()
