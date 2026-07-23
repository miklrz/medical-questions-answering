"""
LangGraph pipeline: Query Analysis → Retrieval → Reranking → Answer Generation → Quality Check.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END

from src.api.config import APIConfig, GraphConfig, LLMConfig, get_api_config
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
    has_urgency: bool


def query_analysis_node(state: GraphState, config: GraphConfig) -> dict:
    """Analyze query: extract intent, detect urgency."""
    logging.info("Start query analysis")
    query = state["user_question"].strip()
    has_urgency = any(kw in query.lower() for kw in config.urgency_keywords)
    return {
        "analyzed_query": query,
        "need_clarification": False,
        "has_urgency": has_urgency,
    }


def retrieval_node(state: GraphState, retrieval: Retrieval, top_k: int) -> dict:
    """Retrieve top-k candidates from FAISS."""
    logging.info("Retrieving top-k candidates from FAISS")
    query = state["analyzed_query"]
    candidates = retrieval.query(query, top_k=top_k)
    return {"retrieved_docs": candidates}


def reranking_node(state: GraphState, similarity_model: SimilarityModel, top_n: int) -> dict:
    """Rerank with BERT, keep top-n."""
    logging.info("Reranking")
    query = state["analyzed_query"]
    docs = state["retrieved_docs"]
    scored = []
    for d in docs:
        score = similarity_model.predict(query, d["text"])
        scored.append({"text": d["text"], "sim": score})
    top_docs = sorted(scored, key=lambda x: x["sim"], reverse=True)[:top_n]
    contexts = "\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(top_docs)])
    return {"reranked_docs": top_docs, "contexts": contexts}


def answer_generation_node(
    state: GraphState,
    system_variant: str,
    llm_config: LLMConfig,
) -> dict:
    """Generate answer via LLM with structured output."""
    logging.info("Generating answer")
    contexts = state["contexts"]
    question = state["user_question"]
    prompt = build_generation_prompt(contexts, question, system_variant)
    sources = [c["text"] for c in state["reranked_docs"]]

    result = generate_structured(prompt, sources, config=llm_config)
    logging.info("Answer generated")
    return {
        "validated_answer": result,
        "confidence": result.confidence,
    }


def quality_check_node(state: GraphState, config: GraphConfig) -> dict:
    """Check quality: low confidence or urgency → force doctor visit."""
    confidence = state.get("confidence", 0.0)
    has_urgency = state.get("has_urgency", False)
    answer = state.get("validated_answer")

    need_clarification = confidence < config.low_confidence_threshold and answer is not None

    if answer:
        if need_clarification:
            answer.warnings = list(answer.warnings or []) + [
                "Уверенность модели в ответе низкая. Рекомендуется уточнить вопрос или обратиться к специалисту."
            ]
            answer.requires_doctor_visit = True

        if has_urgency:
            answer.requires_doctor_visit = True
            warning = "Обнаружены признаки срочного состояния. Обратитесь за медицинской помощью немедленно."
            if warning not in (answer.warnings or []):
                answer.warnings = list(answer.warnings or []) + [warning]

    return {"need_clarification": need_clarification, "validated_answer": answer}


def build_medical_qa_graph(
    retrieval: Retrieval,
    similarity_model: SimilarityModel,
    system_variant: str | None = None,
    config: APIConfig | None = None,
) -> StateGraph:
    """Build LangGraph for Medical QA."""
    app_config = config or get_api_config()
    variant = system_variant or app_config.graph.system_variant

    def _analysis(s: GraphState) -> dict:
        return query_analysis_node(s, app_config.graph)

    def _retrieval(s: GraphState) -> dict:
        return retrieval_node(s, retrieval, app_config.retrieval.top_k)

    def _reranking(s: GraphState) -> dict:
        return reranking_node(s, similarity_model, app_config.reranker.top_n)

    def _generation(s: GraphState) -> dict:
        return answer_generation_node(s, variant, app_config.llm)

    def _quality_check(s: GraphState) -> dict:
        return quality_check_node(s, app_config.graph)

    workflow = StateGraph(GraphState)

    workflow.add_node("query_analysis", _analysis)
    workflow.add_node("retrieval", _retrieval)
    workflow.add_node("reranking", _reranking)
    workflow.add_node("answer_generation", _generation)
    workflow.add_node("quality_check", _quality_check)

    workflow.set_entry_point("query_analysis")
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "reranking")
    workflow.add_edge("reranking", "answer_generation")
    workflow.add_edge("answer_generation", "quality_check")
    workflow.add_edge("quality_check", END)

    return workflow.compile()
