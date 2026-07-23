import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from src.api.config import get_api_config
from src.api.models import SimilarityModel
from src.api.retrieval import Retrieval
from src.api.dataset import MedicalDataset
from src.api.graph import build_medical_qa_graph
from src.api.schemas import MedicalAnswer, Query
from src.api.feedback_store import record_feedback
import logging

logging.basicConfig(level=logging.INFO)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_api_config()

    dataset = MedicalDataset(
        config.dataset.qa_name,
        split=config.dataset.qa_split,
        cache_dir=config.dataset.qa_cache_dir,
    )
    similarity_model = SimilarityModel(
        model_name=config.reranker.model_name,
        device=config.reranker.device,
        SAVED_MODEL_PATH=config.reranker.saved_model_path,
    )
    retrieval = Retrieval(
        embedding_model_name=config.retrieval.embedding_model_name,
        device=config.retrieval.device,
        embedding_batch_size=config.retrieval.embedding_batch_size,
        show_progress_bar=config.retrieval.show_progress_bar,
        embeddings_cache_dir=config.retrieval.embeddings_cache_dir,
        use_embeddings_cache=config.retrieval.use_embeddings_cache,
    )
    retrieval.build_index(dataset.get_qa_pairs())
    graph = build_medical_qa_graph(retrieval, similarity_model, config=config)

    state["graph"] = graph
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/answer")
async def answer(query: Query):
    """Answer medical question via LangGraph pipeline with structured output."""
    logging.info(f"Got request: {query}")
    user_q = query.question
    try:
        result = state["graph"].invoke({"user_question": user_q})
        ans: MedicalAnswer | None = result.get("validated_answer")
        if ans is None:
            raise HTTPException(status_code=500, detail="Failed to generate answer")

        request_id = str(uuid.uuid4())
        return {
            "request_id": request_id,
            "answer": ans.answer,
            "confidence": ans.confidence,
            "sources": ans.sources,
            "requires_doctor_visit": ans.requires_doctor_visit,
            "warnings": ans.warnings,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback(request_id: str, useful: bool):
    """Record human feedback: ?request_id=...&useful=true|false"""
    record_feedback(request_id, useful)
    return {"status": "ok", "request_id": request_id, "useful": useful}


@app.get("/feedback/stats")
async def feedback_stats():
    """Get aggregate feedback statistics."""
    from src.api.feedback_store import get_aggregate_stats

    return get_aggregate_stats()
