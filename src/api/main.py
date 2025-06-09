from fastapi import FastAPI, HTTPException
from src.api.model import SimilarityModel
from src.api.retrieval import Retrieval
from src.api.dataset import MedicalDataset, train_test, tokenize
from src.api.train import ModelTrainer
from src.api.dataset import Query
import requests
from datasets import load_dataset
from langchain_community.chat_models import ChatOpenAI
import os


dataset = MedicalDataset("ruslanmv/ai-medical-chatbot", split="train")
similarity_model = SimilarityModel()
retrieval = Retrieval()
retrieval.build_index(dataset.get_qa_pairs())


def generate_answer_with_llm(contexts: str, user_question: str) -> str:
    prompt = (
        f"You are a helpful and cautious medical assistant. Use the following context to provide an informative answer to the user's question.\n"
        f"Context:\n{contexts}\n\n"
        f"Question: {user_question}\nОтвет:"
    )

    llm = ChatOpenAI(
        temperature=0.0,
        base_url=f"http://{os.getenv("HOST")}:1234/v1",
        api_key="not-needed",
    )
    response = llm.invoke(prompt).content
    return response


app = FastAPI()


@app.post("/answer")
async def answer(query: Query):
    user_q = query.question
    candidates = retrieval.query(user_q, top_k=5)
    scored = []
    for c in candidates:
        score = similarity_model.predict(user_q, c["text"])
        scored.append({"text": c["text"], "sim": score})
    top_n = sorted(scored, key=lambda x: x["sim"], reverse=True)[:5]
    contexts = "\n".join([c["text"] for c in top_n])
    answer = generate_answer_with_llm(contexts, user_q)
    return {"answer": answer, "sources": [c["text"] for c in top_n]}


# if __name__ == "__main__":
#     dataset = MedicalDataset("ruslanmv/ai-medical-chatbot", split="train")
#     similarity_model = SimilarityModel()
#     retrieval = Retrieval()
#     retrieval.build_index(dataset.get_qa_pairs())
