import uuid

from fastapi import FastAPI, HTTPException
from src.api.models import SimilarityModel
from src.api.retrieval import Retrieval
from src.api.dataset import MedicalDataset
from src.api.graph import build_medical_qa_graph
from src.api.schemas import MedicalAnswer, Query
from src.api.feedback_store import record_feedback
import os
from dotenv import load_dotenv

load_dotenv()
dataset = MedicalDataset("ruslanmv/ai-medical-chatbot", split="train")
SAVED_MODEL_PATH = os.getenv("SAVED_MODEL_PATH")
similarity_model = SimilarityModel(SAVED_MODEL_PATH=SAVED_MODEL_PATH)
retrieval = Retrieval()
retrieval.build_index(dataset.get_qa_pairs())

# LangGraph pipeline with full prompt (base + few-shot + CoT)
graph = build_medical_qa_graph(retrieval, similarity_model, system_variant="full")

app = FastAPI()


@app.post("/answer")
async def answer(query: Query):
    """Answer medical question via LangGraph pipeline with structured output."""
    user_q = query.question
    try:
        result = graph.invoke({"user_question": user_q})
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
