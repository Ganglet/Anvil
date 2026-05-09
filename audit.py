"""
ANVIL — Adversarial Neural Vulnerability Inspection and Learning
CLI entry point. All 7 pipeline phases wired.
"""
import argparse
import logging
import sys
from pathlib import Path

import torch

from models.image_model import ImageModel
from models.text_model import TextModel
from profiler.attack_surface import AttackSurfaceProfiler
from attacks.engine import AttackEngine
from clustering.feature_extractor import FeatureExtractor
from clustering.clusterer import FailureModeClusterer
from agent.graph import run_agent
from patching.patcher import Patcher
from reporter.report import generate_report

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_TEXT_EXTS  = {".txt"}

log = logging.getLogger("anvil")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="audit",
        description="ANVIL: adversarial neural vulnerability inspection and learning",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet18", "distilbert"],
        help="Model to audit",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input samples directory (images or .txt files)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./audit_report.pdf",
        help="Output path for the PDF audit report (default: ./audit_report.pdf)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="Number of attack samples to generate (default: 20)",
    )
    return parser.parse_args()


def _load_model(model_arg: str):
    if model_arg == "resnet18":
        return ImageModel(pretrained=True)
    return TextModel()


def _load_image_inputs(input_dir: Path, n: int):
    """Load up to n images from input_dir, apply ImageNet transforms."""
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    files = [p for p in sorted(input_dir.iterdir())
             if p.suffix.lower() in _IMAGE_EXTS][:n]
    if not files:
        return None
    tensors = []
    for f in files:
        try:
            img = Image.open(f).convert("RGB")
            tensors.append(tf(img))
        except Exception:
            log.warning("Skipping unreadable image: %s", f.name)
    if not tensors:
        return None
    return torch.stack(tensors)


def _load_text_inputs(input_dir: Path, n: int):
    """Load up to n .txt files and tokenize for DistilBERT."""
    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    files = [p for p in sorted(input_dir.iterdir())
             if p.suffix.lower() in _TEXT_EXTS][:n]
    if not files:
        return None
    texts = [f.read_text(encoding="utf-8").strip() for f in files]
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=64, return_tensors="pt")
    return enc["input_ids"]


def _make_inputs(model, input_path: str, n: int):
    """
    Load real inputs from disk when available; fall back to synthetic tensors.
    Labels are always derived from the model's own predictions so they are
    consistent ground truth for a model we are probing, not training.
    """
    p = Path(input_path)
    loaded = None

    if p.is_dir():
        if "distilbert" in model.model_name:
            loaded = _load_text_inputs(p, n)
        else:
            loaded = _load_image_inputs(p, n)

    if loaded is not None and len(loaded) > 0:
        inputs = loaded
        log.info("Loaded %d real inputs from %s", len(inputs), input_path)
    else:
        log.warning(
            "No valid input files found in '%s' — using synthetic inputs", input_path
        )
        if "distilbert" in model.model_name:
            inputs = torch.randint(100, 30000, (n, 64))
        else:
            inputs = torch.rand(n, 3, 224, 224)

    with torch.no_grad():
        logits = model.predict(inputs)
    labels = logits.argmax(dim=1).tolist()
    return inputs, labels


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()

    log.info("[ANVIL] Starting audit")
    log.info("  Model  : %s", args.model)
    log.info("  Input  : %s", args.input)
    log.info("  Output : %s", args.output)
    log.info("  Budget : %d samples", args.budget)
    log.info("")

    # ── Phase 1: Load model ───────────────────────────────────────────────────
    model = _load_model(args.model)
    log.info("[1/7] Model loaded: %s", model.model_name)

    inputs, labels = _make_inputs(model, args.input, args.budget)
    log.info("      %d inputs ready", len(inputs))

    # ── Phase 2: Attack surface profiler ──────────────────────────────────────
    profile_n = min(10, args.budget)
    profiler = AttackSurfaceProfiler(model)
    profile = profiler.profile(inputs[:profile_n], labels[:profile_n])

    log.info("[2/7] Profile complete")
    log.info("      Vulnerability score : %.3f", profile["vulnerability_score"])
    log.info("      Attack priority     : %s", profile["attack_priority"][:3])

    # ── Phase 3: Multi-strategy attack engine ─────────────────────────────────
    engine = AttackEngine(model)
    attack_results = engine.run(inputs, labels, profile)
    rates = engine.success_rate(attack_results)

    log.info("[3/7] Attacks complete")
    for name, rate in rates.items():
        log.info("      %-10s %.1f%% success rate", name, rate * 100)

    all_examples = [ex for exs in attack_results.values() for ex in exs]
    total_success = sum(1 for ex in all_examples if ex.success)
    log.info("      %d/%d examples fooled the model", total_success, len(all_examples))

    # ── Phase 4: Failure mode clustering ──────────────────────────────────────
    extractor = FeatureExtractor(model, profile)
    vectors, successful = extractor.extract(all_examples)

    if len(vectors) < 2:
        log.info("[4/7] Clustering skipped — fewer than 2 successful attacks")
        return 0

    taxonomy = FailureModeClusterer().cluster(
        vectors, successful, model_name=model.model_name
    )
    s = taxonomy.summary()
    log.info("[4/7] Clustering complete")
    log.info("      %d vulnerability clusters, %d noise points",
             s["num_clusters"], s["noise_count"])
    for c in s["clusters"]:
        dist = ", ".join(f"{k}:{v}" for k, v in c["attack_distribution"].items())
        log.info("      [%d] %s  size=%d  (%s)", c["id"], c["name"], c["size"], dist)

    # ── Phase 5: LLM explanation agent ────────────────────────────────────────
    log.info("[5/7] Running LLM explanation agent")
    report = run_agent(taxonomy)
    rs = report.summary()
    log.info("      %d clusters explained", rs["num_clusters_explained"])
    for e in report.explanations:
        log.info("      [%d] strategy=%s  sources=%d",
                 e.cluster_id, e.patch_strategy, len(e.sources))
        log.info("      %s...", e.explanation[:120])

    # ── Phase 6: Autonomous patching ──────────────────────────────────────────
    log.info("[6/7] Running autonomous patching")
    patcher = Patcher()
    patch_report = patcher.patch(model, taxonomy, report, inputs, labels)
    ps = patch_report.summary()
    log.info("      %d/%d clusters patched, %d unresolved",
             ps["patched"], ps["total_clusters"], ps["unresolved"])
    for r in patch_report.results:
        status = "pass" if r.passed else "FAIL"
        log.info("      [%s] cluster %d  strategy=%s  score=%.3f  retries=%d",
                 status, r.cluster_id, r.strategy, r.safety_score, r.retries)

    # ── Phase 7: PDF report generation ────────────────────────────────────────
    log.info("[7/7] Generating PDF audit report")
    generate_report(
        output_path=args.output,
        model_name=model.model_name,
        profile=profile,
        attack_rates=rates,
        total_fooled=total_success,
        total_examples=len(all_examples),
        taxonomy=taxonomy,
        explanation_report=report,
        patch_report=patch_report,
    )
    log.info("      Report written to: %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
