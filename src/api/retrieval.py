from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import os


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

    def build_index(self, texts: list, cache_dir: str = None):
        self.corpus = texts

        embeddings_path = (
            os.path.join(cache_dir, "embeddings.npy") if cache_dir else None
        )
        index_path = os.path.join(cache_dir, "faiss.index") if cache_dir else None
        corpus_path = os.path.join(cache_dir, "corpus.npy") if cache_dir else None

        if cache_dir and all(
            os.path.exists(p) for p in [embeddings_path, index_path, corpus_path]
        ):
            print("Loading index from cache")
            self.embeddings = np.load(embeddings_path)
            self.index = faiss.read_index(index_path)
            self.corpus = np.load(corpus_path, allow_pickle=True).tolist()
            print("Index loaded")
            return

        self.embeddings = self.embedder.encode(
            texts, convert_to_numpy=True, batch_size=128, show_progress_bar=True
        )
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)
        print("Index built")

        print(f"Embeddings shape: {self.embeddings.shape}")
        print(f"Embeddings size MB: {self.embeddings.nbytes / 1024**2:.1f}")
        print(f"Corpus len: {len(self.corpus)}")

        if cache_dir:
            np.save(embeddings_path, self.embeddings)
            np.save(corpus_path, self.corpus)
            faiss.write_index(self.index, index_path)
            print("Index cached")

    def query(self, query_text: str, top_k: int = 5):
        q_emb = self.embedder.encode([query_text], convert_to_numpy=True)
        dists, idxs = self.index.search(q_emb, top_k)
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            results.append({"text": self.corpus[idx], "score": float(dist)})
        return results
