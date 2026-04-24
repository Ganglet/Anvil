# Problems Faced & Key Decisions — Red Queen

This document is a running log of every non-trivial problem encountered and every design decision made during the project. Updated after each implementation step. Serves as source material for the README's limitations section and interview answers.

---

## P1

**Phase:** 1 — Scaffold & Model-Agnostic Interface
**Where it surfaced:** `tests/test_models.py::test_text_predict_shape`
**Problem:** Fatal `EXC_ARM_DA_ALIGN` (SIGBUS) crash inside Apple Accelerate BLAS (`libBLAS → SGEMM`) when running DistilBERT forward pass on Apple Silicon (MacBookPro18,1, M1 Pro). ResNet-18 tests passed fine. Crash report showed the faulting address was inside a 255MB memory-mapped file region — the DistilBERT weights loaded by `safetensors`.
**Root cause:** `safetensors` memory-maps weight files directly from disk. Individual tensors within the file can start at byte offsets that are not aligned to the boundary Apple's Accelerate BLAS requires for SIMD (NEON/AMX) operations. ResNet avoids this because torchvision loads `.bin` weights via pickle, which copies data into normally-allocated (aligned) heap memory.
**Decision/Fix:** After `from_pretrained()`, clone every model parameter: `for param in model.parameters(): param.data = param.data.clone()`. This copies all weights from the mmap'd region into fresh heap-allocated memory with proper alignment. Three-line fix, no external dependencies, no re-download.
**Why not alternative:** `use_safetensors=False` would fall back to `.bin` format but requires re-downloading ~260MB. Downgrading torch didn't help — the bug is in Apple's Accelerate BLAS behavior with mmap'd memory, not in PyTorch itself.
