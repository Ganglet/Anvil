"""
Build the FAISS knowledge base from actual paper PDFs.
Run once: python knowledge_base/build_index.py
Produces knowledge_base/faiss_index/ — committed to git.
"""
import pickle
from pathlib import Path

import faiss
import fitz  # pymupdf
import numpy as np
from sentence_transformers import SentenceTransformer

PDFS_DIR = Path(__file__).parent / "pdfs"
INDEX_DIR = Path(__file__).parent / "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Human-readable names for each PDF (arXiv ID → paper name)
PDF_NAMES = {
    "1312.6199v4.pdf": "Szegedy 2013 — Intriguing Properties of Neural Networks",
    "1412.6572v3.pdf": "Goodfellow 2014 — Explaining and Harnessing Adversarial Examples (FGSM)",
    "1511.04508v2.pdf": "Papernot 2016 — Distillation as a Defense to Adversarial Perturbations",
    "1608.04644v2.pdf": "Carlini & Wagner 2017 — Towards Evaluating the Robustness of Neural Networks",
    "1706.06083v4.pdf": "Madry 2018 — Towards Deep Learning Models Resistant to Adversarial Attacks (PGD)",
    "1712.09665v2.pdf": "Brown 2017 — Adversarial Patch",
    "1811.12231v3.pdf": "Geirhos 2019 — ImageNet-Trained CNNs Are Biased Towards Texture",
    "1905.02175v4.pdf": "Ilyas 2019 — Adversarial Examples Are Not Bugs, They Are Features",
    "2004.07780v5.pdf": "Geirhos 2020 — Shortcut Learning in Deep Neural Networks",
    "file.pdf": "Zech 2018 — Variable Generalization of a Deep Learning Model for Pneumonia Detection",
}

CHUNK_SIZE = 400   # characters per chunk
CHUNK_OVERLAP = 80


def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n".join(pages)


def chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks for fine-grained retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"source": source, "text": chunk})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build(model_name: str = MODEL_NAME):
    print(f"Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)

    all_chunks = []
    for pdf_file in sorted(PDFS_DIR.glob("*.pdf")):
        name = PDF_NAMES.get(pdf_file.name, pdf_file.name)
        print(f"  Extracting: {name}")
        text = extract_text(pdf_file)
        chunks = chunk_text(text, source=name)
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Embedding...")

    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalised vecs = cosine similarity
    index.add(embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\nIndex saved to {INDEX_DIR}/")
    print(f"  {index.ntotal} vectors, dim={dim}")


if __name__ == "__main__":
    build()
