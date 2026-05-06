# Phase 5 — LangGraph LLM Explanation Agent

## What this phase does

Phase 4 produced a `VulnerabilityTaxonomy` — named clusters of failure modes. Phase 5 answers the harder question: **what do these clusters mean, and how should they be fixed?**

For each cluster, a LangGraph agent:
1. Retrieves the most relevant paper chunks from the FAISS knowledge base
2. Sends those chunks + cluster metadata to Gemini 2.5 Flash to write a technical explanation
3. Asks the LLM to map the cluster to one of 4 patching strategies with parameters

Output: an `ExplanationReport` — one `ClusterExplanation` per cluster, containing the written explanation, recommended patch strategy, strategy parameters, and source papers cited. Phase 6 reads this to know what to patch and how.

---

## Pipeline

```
VulnerabilityTaxonomy (Phase 4)
        │
        ▼  for each cluster:
┌─────────────────────────────────────────┐
│  LangGraph StateGraph                   │
│                                         │
│  retrieve → explain → recommend         │
│                                         │
│  retrieve:   query FAISS index          │
│              cluster name + attack →    │
│              top-5 paper chunks         │
│                                         │
│  explain:    Gemini 2.5 Flash           │
│              chunks + cluster →         │
│              3-5 sentence explanation   │
│                                         │
│  recommend:  Gemini 2.5 Flash           │
│              explanation + cluster →    │
│              patch strategy + params    │
└─────────────────────────────────────────┘
        │
        ▼
ExplanationReport
  └─ ClusterExplanation × num_clusters
```

---

## Knowledge base

**Source:** 10 adversarial ML papers (actual PDFs, not summaries):
1. Szegedy 2013 — Intriguing Properties of Neural Networks
2. Goodfellow 2014 — FGSM
3. Papernot 2016 — Defensive Distillation
4. Carlini & Wagner 2017 — C&W Attack
5. Madry 2018 — PGD + Adversarial Training
6. Brown 2017 — Adversarial Patch
7. Geirhos 2019 — Texture Bias
8. Ilyas 2019 — Features Not Bugs
9. Geirhos 2020 — Shortcut Learning
10. Zech 2018 — Spurious Correlations in Medical Imaging

**Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (22MB, CPU, ~380-dim vectors)
**Index:** FAISS `IndexFlatIP` (cosine similarity on L2-normalised vectors), 2068 chunks
**Chunking:** 400 character chunks, 80 character overlap

Built once via `python knowledge_base/build_index.py`. Index committed to git.

---

## Key design decisions

### Why LangGraph over a single LLM call
The retrieve → explain → recommend flow has three distinct responsibilities. A single prompt doing all three at once produces less reliable output — the LLM tends to conflate explanation and recommendation. Separating them into LangGraph nodes makes each step's output inspectable and testable independently. It also means Phase 6 gets a clean structured recommendation, not free text to parse.

### Why two LLM calls per cluster (explain + recommend)
The explain node uses retrieved paper chunks as context — it needs the papers to write an accurate explanation. The recommend node uses the explanation as context — it needs to understand the vulnerability before recommending a fix. Combining them forces the LLM to both explain and recommend from paper context in one prompt, which degrades both outputs. Two focused calls produce better results.

### Why `all-MiniLM-L6-v2` for embeddings (see D3 in problems_and_decisions.md)
22MB, runs on CPU in milliseconds, no server required. Original plan was `nomic-embed-text` via Ollama — dropped because it adds server infrastructure for no quality benefit at 10-paper scale.

### Why Gemini 2.5 Flash (see D2 in problems_and_decisions.md)
Originally Groq (Llama 3 70B) — dropped due to account creation failure. Gemini 2.5 Flash is free, drop-in compatible via LangChain, and has a 1M token context window.

### Structured recommendation format
The recommend node instructs the LLM to respond in a fixed format (`STRATEGY: ...`, `PARAM_LAYERS: ...`, etc.) rather than free text. `_parse_recommendation()` is a simple line parser with safe fallbacks for every field — if the LLM deviates, the system falls back to sensible defaults rather than crashing. This is important because LLM output format compliance is not guaranteed.

---

## The 4 patch strategies

Phase 5 maps each cluster to one of these. Phase 6 executes them.

| Strategy | Best for |
|----------|----------|
| `adversarial_training` | Gradient sensitivity (FGSM/PGD clusters) |
| `stylized_augmentation` | Texture bias / semantic sensitivity |
| `counterfactual_generation` | Background dependency / shortcut learning |
| `targeted_augmentation` | Edge-case underrepresentation |

---

## Files

| File | Role |
|------|------|
| `agent/schema.py` | `ClusterExplanation` + `ExplanationReport` dataclasses |
| `agent/retriever.py` | FAISS query — cluster description → top-k paper chunks |
| `agent/nodes.py` | Three LangGraph nodes: retrieve, explain, recommend |
| `agent/graph.py` | Wires nodes into a compiled StateGraph; `run_agent()` entry point |
| `agent/__init__.py` | Exports |
| `knowledge_base/build_index.py` | One-time script to embed PDFs and write FAISS index |
| `knowledge_base/pdfs/` | 10 source paper PDFs |
| `knowledge_base/faiss_index/` | `index.faiss` + `chunks.pkl` — committed to git |

---

## Test results

```
tests/test_agent.py — 15/15 passed
```

| Test group | Coverage |
|------------|----------|
| Schema | ClusterExplanation fields, ExplanationReport summary, empty report |
| `_parse_recommendation` | Valid input, invalid strategy fallback, invalid steps fallback, all 4 strategies |
| PaperRetriever | top-k count, result keys, score range, named sources |
| LangGraph nodes (mocked LLM) | retrieve populates chunks, explain sets explanation, recommend returns ClusterExplanation, source deduplication |

LLM is mocked in all node tests — no API calls during `pytest`. Real Gemini calls only happen via `audit.py`.

---

## Connection to adjacent phases

- **Receives from Phase 4:** `VulnerabilityTaxonomy` — cluster names, sizes, dominant attacks
- **Passes to Phase 6:** `ExplanationReport` — patch strategy + parameters per cluster, which Phase 6 executes
- **Passes to Phase 7:** `ExplanationReport` — full explanation text rendered into the PDF audit report
