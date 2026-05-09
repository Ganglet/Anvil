"""
ANVIL FastAPI server — triggers audit pipeline via HTTP.
Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""
import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel as PydanticModel, Field

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

log = logging.getLogger("anvil.api")
logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(
    title="ANVIL",
    description="Adversarial Neural Vulnerability Inspection and Learning — REST API",
    version="1.0.0",
)


class AuditRequest(PydanticModel):
    model: str = Field("resnet18", pattern="^(resnet18|distilbert)$",
                        description="Model to audit")
    budget: int = Field(20, ge=5, le=200,
                         description="Number of attack samples (5–200)")


class AuditStatus(PydanticModel):
    status: str
    model: str
    budget: int
    vulnerability_score: float | None = None
    clusters_found: int | None = None
    clusters_patched: int | None = None
    report_path: str | None = None
    error: str | None = None


def _load_model(model_arg: str):
    if model_arg == "resnet18":
        return ImageModel(pretrained=True)
    return TextModel()


def _synthetic_inputs(model, n: int):
    if "distilbert" in model.model_name:
        inputs = torch.randint(100, 30000, (n, 64))
    else:
        inputs = torch.rand(n, 3, 224, 224)
    with torch.no_grad():
        logits = model.predict(inputs)
    labels = logits.argmax(dim=1).tolist()
    return inputs, labels


@app.get("/health")
def health():
    return {"status": "ok", "service": "ANVIL"}


@app.post("/audit", response_model=AuditStatus)
def run_audit(req: AuditRequest):
    log.info("Audit requested: model=%s budget=%d", req.model, req.budget)
    try:
        model = _load_model(req.model)
        inputs, labels = _synthetic_inputs(model, req.budget)

        profile_n = min(10, req.budget)
        profiler = AttackSurfaceProfiler(model)
        profile = profiler.profile(inputs[:profile_n], labels[:profile_n])

        engine = AttackEngine(model)
        attack_results = engine.run(inputs, labels, profile)
        rates = engine.success_rate(attack_results)
        all_examples = [ex for exs in attack_results.values() for ex in exs]
        total_success = sum(1 for ex in all_examples if ex.success)

        vectors, successful = FeatureExtractor(model, profile).extract(all_examples)
        if len(vectors) < 2:
            raise HTTPException(
                status_code=422,
                detail="Fewer than 2 successful attacks — increase budget or try a different model.",
            )

        taxonomy = FailureModeClusterer().cluster(
            vectors, successful, model_name=model.model_name
        )
        ts = taxonomy.summary()

        explanation_report = run_agent(taxonomy)

        patcher = Patcher()
        patch_report = patcher.patch(model, taxonomy, explanation_report, inputs, labels)
        ps = patch_report.summary()

        out_dir = Path(os.getenv("ANVIL_OUTPUT_DIR", "/tmp/anvil_reports"))
        out_dir.mkdir(parents=True, exist_ok=True)
        import datetime
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"audit_{req.model}_{ts_str}.pdf"

        generate_report(
            output_path=str(out_path),
            model_name=model.model_name,
            profile=profile,
            attack_rates=rates,
            total_fooled=total_success,
            total_examples=len(all_examples),
            taxonomy=taxonomy,
            explanation_report=explanation_report,
            patch_report=patch_report,
        )

        log.info("Audit complete — report: %s", out_path)
        return AuditStatus(
            status="complete",
            model=req.model,
            budget=req.budget,
            vulnerability_score=profile["vulnerability_score"],
            clusters_found=ts["num_clusters"],
            clusters_patched=ps["patched"],
            report_path=str(out_path),
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Audit failed")
        return AuditStatus(
            status="error",
            model=req.model,
            budget=req.budget,
            error=str(exc),
        )


@app.get("/report/{filename}")
def download_report(filename: str):
    out_dir = Path(os.getenv("ANVIL_OUTPUT_DIR", "/tmp/anvil_reports"))
    path = out_dir / filename
    if not path.exists() or path.suffix != ".pdf":
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(str(path), media_type="application/pdf", filename=filename)
