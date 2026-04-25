"""
Red Queen — Autonomous ML Red-Teaming Agent
CLI entry point. Phases 2-7 will fill in the pipeline steps.
"""
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        prog="audit",
        description="Red Queen: autonomous ML red-teaming agent",
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
        default=100,
        help="Number of attack samples to generate (default: 100)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[Red Queen] Starting audit")
    print(f"  Model  : {args.model}")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Budget : {args.budget} samples")
    print()

    # Phase 2: Attack surface profiler         (coming in Phase 2)
    # Phase 3: Multi-strategy attack engine    (coming in Phase 3)
    # Phase 4: Failure mode clustering         (coming in Phase 4)
    # Phase 5: LLM explanation agent           (coming in Phase 5)
    # Phase 6: Autonomous patching             (coming in Phase 6)
    # Phase 7: PDF report generation           (coming in Phase 7)

    print("[Red Queen] Pipeline not yet implemented — scaffold only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
