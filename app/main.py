from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from . import schemas
    from .analysis_builder import (
        build_forensic_outputs,
        build_global_summary,
        build_xai_quality_summary,
        get_or_create_explanation,
    )
    from .analysis_runtime import analysis_store
    from .ml_service import InferenceError, ml_engine
    from .xai_service import xai_engine
except ImportError:
    import schemas
    from analysis_builder import (
        build_forensic_outputs,
        build_global_summary,
        build_xai_quality_summary,
        get_or_create_explanation,
    )
    from analysis_runtime import analysis_store
    from ml_service import InferenceError, ml_engine
    from xai_service import xai_engine


BACKEND_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BACKEND_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="X-GRU NIDS API",
    description="Inference and forensic explanation API for the final FYP pipeline",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExplainRequest(BaseModel):
    window_id: int


def ensure_current_analysis_ready() -> None:
    if analysis_store.detection_response is None:
        raise HTTPException(
            status_code=400,
            detail="No current upload analysis found. Upload a CSV first.",
        )


@app.get("/")
def health_check():
    return {
        "status": "online",
        "message": "FYP intrusion detection API is ready.",
        "model_ready": ml_engine.is_ready,
        "current_analysis_ready": analysis_store.detection_response is not None,
        "current_filename": analysis_store.filename,
    }


@app.get("/model-info", response_model=schemas.ModelInfoResponse)
def model_info():
    return schemas.ModelInfoResponse(
        class_names=ml_engine.class_names,
        sequence_length=ml_engine.sequence_length,
        feature_count=len(ml_engine.feature_columns),
        feature_columns=ml_engine.feature_columns,
    )


@app.post("/upload-traffic", response_model=schemas.DetectionResponse)
async def analyze_traffic(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        contents = await file.read()
        upload_path = UPLOAD_DIR / Path(file.filename).name
        upload_path.write_bytes(contents)
        dataframe = pd.read_csv(pd.io.common.BytesIO(contents), low_memory=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV file: {exc}") from exc

    try:
        response_payload = ml_engine.predict_traffic(dataframe, filename=file.filename)
    except InferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Traffic analysis failed: {exc}") from exc

    background_sequences = ml_engine.get_background_sequences(max_samples=50)
    if background_sequences is not None:
        try:
            xai_engine.prepare(
                model=ml_engine.model,
                background_sequences=background_sequences,
                feature_columns=ml_engine.feature_columns,
                sequence_length=ml_engine.sequence_length,
                class_names=ml_engine.class_names,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"XAI engine preparation failed: {exc}") from exc

    analysis_store.clear()
    analysis_store.filename = file.filename
    analysis_store.detection_response = response_payload

    try:
        analysis_store.global_summary = build_global_summary(top_n=15)
        analysis_store.xai_quality_summary = build_xai_quality_summary()
        forensic_summary_payload, forensic_cases_payload = build_forensic_outputs()
        analysis_store.forensic_summary = forensic_summary_payload
        analysis_store.forensic_cases = forensic_cases_payload
    except Exception:
        # Keep upload usable even if summary-generation fails.
        analysis_store.global_summary = {"flat_features": [], "base_features": []}
        analysis_store.xai_quality_summary = {
            "mean_jaccard_similarity": None,
            "mean_shap_fidelity": None,
            "mean_lime_fidelity": None,
            "mean_shap_stability": None,
            "mean_lime_stability": None,
            "explained_case_count": 0,
        }
        analysis_store.forensic_summary = {
            "forensic_case_count": 0,
            "label_counts": {},
            "top_shared_indicators": [],
            "mean_jaccard_similarity": None,
            "mean_shap_fidelity": None,
            "mean_lime_fidelity": None,
            "mean_shap_stability": None,
            "mean_lime_stability": None,
        }
        analysis_store.forensic_cases = []

    return schemas.DetectionResponse(**response_payload)


@app.post("/explain-alert", response_model=schemas.XAIResponse)
def explain_alert(request: ExplainRequest):
    ensure_current_analysis_ready()

    try:
        explanation = get_or_create_explanation(request.window_id)
    except InferenceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc

    return schemas.XAIResponse(**explanation)


@app.get("/xai/global-summary", response_model=schemas.GlobalXAIResponse)
def xai_global_summary(top_n: int = Query(default=15, ge=5, le=50)):
    ensure_current_analysis_ready()

    try:
        payload = build_global_summary(top_n=top_n)
        analysis_store.global_summary = payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Global summary generation failed: {exc}") from exc

    return schemas.GlobalXAIResponse(
        flat_features=[
            schemas.FeatureImportanceItem(
                feature_name=str(item["feature_name"]),
                score=float(item["score"]),
            )
            for item in payload.get("flat_features", [])[:top_n]
        ],
        base_features=[
            schemas.FeatureImportanceItem(
                feature_name=str(item["feature_name"]),
                score=float(item["score"]),
            )
            for item in payload.get("base_features", [])[:top_n]
        ],
    )


@app.get("/xai/quality-summary", response_model=schemas.XAIQualitySummaryResponse)
def xai_quality_summary():
    ensure_current_analysis_ready()

    try:
        payload = build_xai_quality_summary()
        analysis_store.xai_quality_summary = payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"XAI quality summary generation failed: {exc}") from exc

    return schemas.XAIQualitySummaryResponse(**payload)


@app.get("/forensic/summary", response_model=schemas.ForensicSummaryResponse)
def forensic_summary():
    ensure_current_analysis_ready()

    try:
        summary_payload, cases_payload = build_forensic_outputs()
        analysis_store.forensic_summary = summary_payload
        analysis_store.forensic_cases = cases_payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forensic summary generation failed: {exc}") from exc

    return schemas.ForensicSummaryResponse(**summary_payload)


@app.get("/forensic/cases", response_model=schemas.ForensicCaseListResponse)
def forensic_cases(
    limit: int = Query(default=50, ge=1, le=500),
    label: str | None = Query(default=None),
):
    ensure_current_analysis_ready()

    try:
        if analysis_store.forensic_cases:
            working_cases = list(analysis_store.forensic_cases)
        else:
            _, working_cases = build_forensic_outputs()
            analysis_store.forensic_cases = working_cases
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forensic case generation failed: {exc}") from exc

    if label:
        working_cases = [
            case
            for case in working_cases
            if str(case.get("predicted_label_name", "")).lower() == label.lower()
        ]

    return schemas.ForensicCaseListResponse(
        total_cases=len(analysis_store.forensic_cases),
        cases=[schemas.ForensicCaseItem(**case) for case in working_cases[:limit]],
    )