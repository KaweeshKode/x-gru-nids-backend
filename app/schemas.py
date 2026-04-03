from typing import Dict, List, Optional

from pydantic import BaseModel


class WindowResult(BaseModel):
    window_id: int
    last_row_id: int
    predicted_label_id: int
    predicted_label_name: str
    probability_normal: float
    probability_suspicious: float
    probability_attack: float
    alert_score: float


class DetectionSummary(BaseModel):
    total_rows: int
    total_windows: int
    suspicious_or_attack_windows: int
    label_counts: Dict[str, int]


class DetectionResponse(BaseModel):
    filename: str
    summary: DetectionSummary
    results: List[WindowResult]


class FeatureWeight(BaseModel):
    feature_name: str
    weight: float


class XAIResponse(BaseModel):
    window_id: int
    predicted_label_name: str
    lime_weights: List[FeatureWeight]
    shap_weights: List[FeatureWeight]


class ModelInfoResponse(BaseModel):
    class_names: List[str]
    sequence_length: int
    feature_count: int
    feature_columns: List[str]


class FeatureImportanceItem(BaseModel):
    feature_name: str
    score: float


class GlobalXAIResponse(BaseModel):
    flat_features: List[FeatureImportanceItem]
    base_features: List[FeatureImportanceItem]


class XAIQualitySummaryResponse(BaseModel):
    mean_jaccard_similarity: Optional[float] = None
    mean_shap_fidelity: Optional[float] = None
    mean_lime_fidelity: Optional[float] = None
    mean_shap_stability: Optional[float] = None
    mean_lime_stability: Optional[float] = None
    explained_case_count: int


class ForensicSummaryResponse(BaseModel):
    forensic_case_count: int
    label_counts: Dict[str, int]
    top_shared_indicators: List[str]
    mean_jaccard_similarity: Optional[float] = None
    mean_shap_fidelity: Optional[float] = None
    mean_lime_fidelity: Optional[float] = None
    mean_shap_stability: Optional[float] = None
    mean_lime_stability: Optional[float] = None


class ForensicCaseItem(BaseModel):
    case_id: str
    sample_row_index: int
    predicted_label_name: str
    probability_attack: float
    probability_suspicious: float
    jaccard_similarity: Optional[float] = None
    shap_fidelity: Optional[float] = None
    lime_fidelity: Optional[float] = None
    shap_stability: Optional[float] = None
    lime_stability: Optional[float] = None
    explanation_quality_summary: str
    plain_language_explanation: str
    analyst_recommendation: str


class ForensicCaseListResponse(BaseModel):
    total_cases: int
    cases: List[ForensicCaseItem]