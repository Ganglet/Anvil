"""
ANVIL — Adversarial Neural Vulnerability Inspection and Learning
CLI entry point. Phases 1-5 wired; 6-7 stubs pending implementation.
"""
import argparse
import sys
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
        help="Path to input samples directory or file",
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


def _make_inputs(model, n: int):
    """
    Synthetic inputs for the pipeline until Phase 8 adds a real input loader.
    Labels are the model's own predictions on the synthetic inputs — consistent
    ground truth for a model we're probing, not training.
    """
    if "distilbert" in model.model_name:
        inputs = torch.randint(100, 30000, (n, 64))
    else:
        inputs = torch.rand(n, 3, 224, 224)

    with torch.no_grad():
        logits = model.predict(inputs)
    labels = logits.argmax(dim=1).tolist()
    return inputs, labels


def main():
    args = parse_args()

    print("[ANVIL] Starting audit")
    print(f"  Model  : {args.model}")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Budget : {args.budget} samples")
    print()

    # ── Phase 1: Load model ───────────────────────────────────────────────────
    model = _load_model(args.model)
    print(f"[1/7] Model loaded: {model.model_name}")

    inputs, labels = _make_inputs(model, args.budget)
    print(f"      {len(inputs)} synthetic inputs  (real loader: Phase 8)")

    # ── Phase 2: Attack surface profiler ──────────────────────────────────────
    profile_n = min(10, args.budget)
    profiler = AttackSurfaceProfiler(model)
    profile = profiler.profile(inputs[:profile_n], labels[:profile_n])

    print(f"[2/7] Profile complete")
    print(f"      Vulnerability score : {profile['vulnerability_score']:.3f}")
    print(f"      Attack priority     : {profile['attack_priority'][:3]}")

    # ── Phase 3: Multi-strategy attack engine ─────────────────────────────────
    engine = AttackEngine(model)
    attack_results = engine.run(inputs, labels, profile)
    rates = engine.success_rate(attack_results)

    print(f"[3/7] Attacks complete")
    for name, rate in rates.items():
        print(f"      {name:<10} {rate:.1%} success rate")

    all_examples = [ex for exs in attack_results.values() for ex in exs]
    total_success = sum(1 for ex in all_examples if ex.success)
    print(f"      {total_success}/{len(all_examples)} examples fooled the model")

    # ── Phase 4: Failure mode clustering ──────────────────────────────────────
    extractor = FeatureExtractor(model, profile)
    vectors, successful = extractor.extract(all_examples)

    if len(vectors) < 2:
        print(f"[4/7] Clustering skipped — fewer than 2 successful attacks")
        return 0

    taxonomy = FailureModeClusterer().cluster(
        vectors, successful, model_name=model.model_name
    )
    s = taxonomy.summary()
    print(f"[4/7] Clustering complete")
    print(f"      {s['num_clusters']} vulnerability clusters, {s['noise_count']} noise points")
    for c in s["clusters"]:
        dist = ", ".join(f"{k}:{v}" for k, v in c["attack_distribution"].items())
        print(f"      [{c['id']}] {c['name']}  size={c['size']}  ({dist})")

    # ── Phase 5: LLM explanation agent ───────────────────────────────────────
    print(f"[5/7] Running LLM explanation agent")
    report = run_agent(taxonomy)
    rs = report.summary()
    print(f"      {rs['num_clusters_explained']} clusters explained")
    for e in report.explanations:
        print(f"      [{e.cluster_id}] strategy={e.patch_strategy}  sources={len(e.sources)}")
        print(f"      {e.explanation[:120]}...")

    # ── Phase 6: Autonomous patching ──────────────────────────────────────────
    print(f"[6/7] Running autonomous patching")
    patcher = Patcher()
    patch_report = patcher.patch(model, taxonomy, report, inputs, labels)
    ps = patch_report.summary()
    print(f"      {ps['patched']}/{ps['total_clusters']} clusters patched, {ps['unresolved']} unresolved")
    for r in patch_report.results:
        status = "pass" if r.passed else "FAIL"
        print(
            f"      [{status}] cluster {r.cluster_id}  strategy={r.strategy}"
            f"  score={r.safety_score:.3f}  retries={r.retries}"
        )

    # ── Phase 7: PDF report generation ───────────────────────────────────────
    print(f"[7/7] Generating PDF audit report")
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
    print(f"      Report written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
