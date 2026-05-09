"""
ANVIL FastAPI server — triggers audit pipeline via HTTP.
Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""
import datetime
import logging
import os
import uuid
from io import BytesIO
from pathlib import Path

import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel as PydanticModel, Field
from torchvision import transforms

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ganglet.github.io",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_jobs: dict[str, dict] = {}

OUT_DIR = Path(os.getenv("ANVIL_OUTPUT_DIR", "/tmp/anvil_reports"))

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class AuditRequest(PydanticModel):
    model: str = Field("resnet18", pattern="^(resnet18|distilbert)$")
    budget: int = Field(20, ge=5, le=200)


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


def _images_from_bytes(file_contents: list[tuple[str, bytes]], n: int):
    tensors = []
    for _, data in file_contents[:n]:
        try:
            img = Image.open(BytesIO(data)).convert("RGB")
            tensors.append(IMAGENET_TRANSFORM(img))
        except Exception:
            continue
    if not tensors:
        return None
    return torch.stack(tensors)


def _run_pipeline(model, inputs, labels, budget: int, model_arg: str):
    profile_n = min(10, budget)
    profiler = AttackSurfaceProfiler(model)
    profile = profiler.profile(inputs[:profile_n], labels[:profile_n])

    engine = AttackEngine(model)
    attack_results = engine.run(inputs, labels, profile)
    rates = engine.success_rate(attack_results)
    all_examples = [ex for exs in attack_results.values() for ex in exs]
    total_success = sum(1 for ex in all_examples if ex.success)

    vectors, successful = FeatureExtractor(model, profile).extract(all_examples)
    if len(vectors) < 2:
        raise ValueError("Fewer than 2 successful attacks — try a larger budget.")

    taxonomy = FailureModeClusterer().cluster(vectors, successful, model_name=model.model_name)
    explanation_report = run_agent(taxonomy)
    patch_report = Patcher().patch(model, taxonomy, explanation_report, inputs, labels)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"audit_{model_arg}_{ts_str}.pdf"

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
    return profile, rates, total_success, taxonomy, explanation_report, patch_report, out_path


def _run_audit_job(job_id: str, file_contents: list, model_arg: str, budget: int):
    _jobs[job_id]["status"] = "running"
    try:
        model = _load_model(model_arg)

        if file_contents and model_arg == "resnet18":
            inputs = _images_from_bytes(file_contents, budget)
            if inputs is None:
                inputs, labels = _synthetic_inputs(model, budget)
            else:
                with torch.no_grad():
                    logits = model.predict(inputs)
                labels = logits.argmax(dim=1).tolist()
        else:
            inputs, labels = _synthetic_inputs(model, budget)

        profile, _, _, taxonomy, _, patch_report, out_path = \
            _run_pipeline(model, inputs, labels, budget, model_arg)

        ts = taxonomy.summary()
        ps = patch_report.summary()

        _jobs[job_id].update({
            "status": "complete",
            "vulnerability_score": round(float(profile["vulnerability_score"]), 3),
            "clusters_found": ts["num_clusters"],
            "clusters_patched": ps["patched"],
            "report_filename": out_path.name,
        })
        log.info("Job %s complete — %s", job_id, out_path.name)
    except Exception as exc:
        log.exception("Job %s failed", job_id)
        _jobs[job_id].update({"status": "error", "error": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok", "service": "ANVIL"}


@app.post("/audit", response_model=AuditStatus)
def run_audit(req: AuditRequest):
    log.info("Audit requested: model=%s budget=%d", req.model, req.budget)
    try:
        model = _load_model(req.model)
        inputs, labels = _synthetic_inputs(model, req.budget)
        profile, rates, total_success, taxonomy, explanation_report, patch_report, out_path = \
            _run_pipeline(model, inputs, labels, req.budget, req.model)
        ts = taxonomy.summary()
        ps = patch_report.summary()
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
        return AuditStatus(status="error", model=req.model, budget=req.budget, error=str(exc))


@app.post("/audit/upload")
async def upload_audit(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(default=[]),
    model: str = Form("resnet18"),
    budget: int = Form(20),
):
    if model not in ("resnet18", "distilbert"):
        raise HTTPException(400, "model must be resnet18 or distilbert")
    if not 5 <= budget <= 50:
        raise HTTPException(400, "budget must be 5–50 for the demo")

    file_contents = []
    for f in files:
        contents = await f.read()
        file_contents.append((f.filename, contents))

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "model": model, "budget": budget}
    background_tasks.add_task(_run_audit_job, job_id, file_contents, model, budget)
    log.info("Job %s queued: model=%s budget=%d files=%d", job_id, model, budget, len(file_contents))
    return {"job_id": job_id}


@app.get("/audit/job/{job_id}")
def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return _jobs[job_id]


@app.get("/report/{filename}")
def download_report(filename: str):
    path = OUT_DIR / filename
    if not path.exists() or path.suffix != ".pdf":
        raise HTTPException(404, "Report not found")
    return FileResponse(str(path), media_type="application/pdf", filename=filename)
