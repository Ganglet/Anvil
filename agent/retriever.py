import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent.parent / "knowledge_base" / "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class PaperRetriever:
    """
    Queries the FAISS index built from the 10 adversarial ML papers.
    Given a cluster description, returns the top-k most relevant paper chunks.
    """

    def __init__(self, top_k: int = 5):
        self._top_k = top_k
        self._embedder = SentenceTransformer(MODEL_NAME)
        self._index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
        with open(INDEX_DIR / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)

    def retrieve(self, query: str) -> List[dict]:
        """
        Args:
            query: free-text description of the vulnerability cluster

        Returns:
            List of dicts with keys: source, text, score
        """
        vec = self._embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, self._top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            results.append({
                "source": chunk["source"],
                "text": chunk["text"],
                "score": float(score),
            })
        return results
