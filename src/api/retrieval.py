from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import logging


class Retrieval:
    def __init__(
        self, embedding_model_name="paraphrase-MiniLM-L3-v2", device=None, corpus=None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(embedding_model_name, device=device)
        self.corpus = corpus or []  # список строк вопросов и ответов
        self.index = None
        self.embeddings = None

    def build_index(self, texts: list):
        self.corpus = texts
        self.embeddings = self.embedder.encode(
            texts, convert_to_numpy=True, batch_size=128, show_progress_bar=True
        )
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)
        logging.info("Index built")

    def query(self, query_text: str, top_k: int = 5):
        q_emb = self.embedder.encode([query_text], convert_to_numpy=True)
        dists, idxs = self.index.search(q_emb, top_k)
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            results.append({"text": self.corpus[idx], "score": float(dist)})
        return results
