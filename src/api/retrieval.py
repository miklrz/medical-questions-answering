from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Retrieval:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", corpus=None):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.corpus = corpus or []  # список строк вопросов и ответов
        self.index = None
        self.embeddings = None

    def build_index(self, texts: list):
        self.corpus = texts
        self.embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def query(self, query_text: str, top_k: int = 5):
        q_emb = self.embedder.encode([query_text], convert_to_numpy=True)
        dists, idxs = self.index.search(q_emb, top_k)
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            results.append({"text": self.corpus[idx], "score": float(dist)})
        return results