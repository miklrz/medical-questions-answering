import hashlib
import json
import logging
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
import torch


class Retrieval:
    def __init__(
        self,
        embedding_model_name="paraphrase-MiniLM-L3-v2",
        device=None,
        corpus=None,
        embedding_batch_size: int = 128,
        show_progress_bar: bool = True,
        embeddings_cache_dir: str | None = None,
        use_embeddings_cache: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model_name = embedding_model_name
        self.embedder = SentenceTransformer(embedding_model_name, device=device)
        self.corpus = corpus or []
        self.index = None
        self.embedding_batch_size = embedding_batch_size
        self.show_progress_bar = show_progress_bar
        self.embeddings_cache_dir = Path(embeddings_cache_dir) if embeddings_cache_dir else None
        self.use_embeddings_cache = use_embeddings_cache

    def build_index(self, texts: list[str]):
        self.corpus = texts
        if self.use_embeddings_cache and self._load_cached_index(texts):
            return

        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            batch_size=self.embedding_batch_size,
            show_progress_bar=self.show_progress_bar,
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        logging.info("Index built")

        if self.use_embeddings_cache:
            self._save_cached_index(texts)

    def query(self, query_text: str, top_k: int = 5):
        if self.index is None:
            raise RuntimeError("Retrieval index is not built")
        q_emb = self.embedder.encode([query_text], convert_to_numpy=True)
        dists, idxs = self.index.search(q_emb, top_k)
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            results.append({"text": self.corpus[idx], "score": float(dist)})
        return results

    def _cache_paths(self) -> tuple[Path, Path, Path] | None:
        if self.embeddings_cache_dir is None:
            return None
        return (
            self.embeddings_cache_dir / "index.faiss",
            self.embeddings_cache_dir / "corpus.json",
            self.embeddings_cache_dir / "metadata.json",
        )

    def _load_cached_index(self, texts: list[str]) -> bool:
        paths = self._cache_paths()
        if paths is None:
            return False
        index_path, corpus_path, metadata_path = paths
        if not index_path.exists() or not corpus_path.exists() or not metadata_path.exists():
            return False

        try:
            with metadata_path.open(encoding="utf-8") as f:
                metadata = json.load(f)
            expected = self._build_metadata(texts)
            if metadata != expected:
                logging.info("Embeddings cache metadata mismatch; rebuilding index")
                return False

            self.index = faiss.read_index(str(index_path))
            with corpus_path.open(encoding="utf-8") as f:
                self.corpus = json.load(f)
            if len(self.corpus) != len(texts):
                logging.info("Embeddings cache corpus size mismatch; rebuilding index")
                self.index = None
                return False
            logging.info("Loaded FAISS index from %s", index_path)
            return True
        except Exception as e:
            logging.info("Failed to load embeddings cache: %s. Rebuilding index", e)
            self.index = None
            return False

    def _save_cached_index(self, texts: list[str]) -> None:
        paths = self._cache_paths()
        if paths is None or self.index is None:
            return
        index_path, corpus_path, metadata_path = paths
        self.embeddings_cache_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with corpus_path.open("w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self._build_metadata(texts), f, ensure_ascii=False, indent=2)
        logging.info("Saved FAISS index to %s", index_path)

    def _build_metadata(self, texts: list[str]) -> dict:
        return {
            "embedding_model_name": self.embedding_model_name,
            "count": len(texts),
            "fingerprint": self._fingerprint(texts),
        }

    @staticmethod
    def _fingerprint(texts: list[str]) -> str:
        digest = hashlib.sha256()
        digest.update(str(len(texts)).encode("utf-8"))
        for text in texts[:100]:
            digest.update(text.encode("utf-8"))
            digest.update(b"\0")
        for text in texts[-100:]:
            digest.update(text.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()
