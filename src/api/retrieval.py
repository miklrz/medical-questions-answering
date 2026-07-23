from sentence_transformers import SentenceTransformer
import faiss
import torch
import logging


class Retrieval:
    def __init__(
        self,
        embedding_model_name="paraphrase-MiniLM-L3-v2",
        device=None,
        corpus=None,
        embedding_batch_size: int = 128,
        show_progress_bar: bool = True,
        embeddings_cache_dir: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(embedding_model_name, device=device)
        self.corpus = corpus or []
        self.index = None
        self.embeddings = None
        self.embedding_batch_size = embedding_batch_size
        self.show_progress_bar = show_progress_bar
        self.embeddings_cache_dir = embeddings_cache_dir

    def build_index(self, texts: list):
        self.corpus = texts
        self.embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            batch_size=self.embedding_batch_size,
            show_progress_bar=self.show_progress_bar,
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
