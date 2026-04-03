from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .analysis_runtime import analysis_store
from .ml_service import ml_engine, InferenceError
from .xai_service import xai_engine


def _top_feature_names(items: list[dict[str, Any]], top_k: int = 5) -> list[str]:
    return [str(item["feature_name"]) for item in items[:top_k] if "feature_name" in item]


def _jaccard_similarity(left: list[str], right: list[str]) -> float | None:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return None
    return float(len(left_set & right_set) / len(union))


def _simple_fidelity(weights: list[dict[str, Any]]) -> float | None:
    if not weights:
        return None
    values = [abs(float(item.get("weight", 0.0))) for item in weights[:5]]
    total = sum(abs(float(item.get("weight", 0.0))) for item in weights)
    if total <= 0:
        return None
    return float(sum(values) / total)


def _simple_stability(weights: list[dict[str, Any]]) -> float | None:
    if not weights:
        return None
    values = [abs(float(item.get("weight", 0.0))) for item in weights[:5]]
    if len(values) < 2:
        return 1.0
    spread = np.std(values)
    mean_val = np.mean(values)
    if mean_val <= 0:
        return None
    score = 1.0 - min(float(spread / (mean_val + 1e-9)), 1.0)
    return max(0.0, score)


def select_target_window_ids(max_attack: int = 25, max_suspicious: int = 25) -> list[int]:
    if analysis_store.detection_response is None:
        raise InferenceError("No current upload analysis found. Upload a CSV first.")

    results = analysis_store.detection_response.get("results", [])

    attack_rows = sorted(
        [row for row in results if row.get("predicted_label_name") == "attack"],
        key=lambda row: float(row.get("probability_attack", 0.0)),
        reverse=True,
    )[:max_attack]

    suspicious_rows = sorted(
        [row for row in results if row.get("predicted_label_name") == "suspicious"],
        key=lambda row: float(row.get("alert_score", 0.0)),
        reverse=True,
    )[:max_suspicious]

    selected = attack_rows + suspicious_rows
    return [int(row["window_id"]) for row in selected]


def get_or_create_explanation(window_id: int) -> dict[str, Any]:
    if window_id in analysis_store.explanation_cache:
        return analysis_store.explanation_cache[window_id]

    instance_3d, predicted_label_id, predicted_label_name = ml_engine.get_sequence_for_explanation(window_id)
    if predicted_label_name == "normal":
        raise InferenceError(f"Window {window_id} is normal and is not eligible for explanation.")

    explanation = xai_engine.explain_instance(
        instance_3d=instance_3d,
        predicted_label_id=predicted_label_id,
        predicted_label_name=predicted_label_name,
        window_id=window_id,
    )
    analysis_store.explanation_cache[window_id] = explanation
    return explanation


def build_global_summary(top_n: int = 15) -> dict[str, Any]:
    base_scores = defaultdict(float)

    for window_id in select_target_window_ids():
        explanation = get_or_create_explanation(window_id)

        for item in explanation.get("shap_weights", []):
            base_scores[str(item["feature_name"])] += abs(float(item.get("weight", 0.0)))

        for item in explanation.get("lime_weights", []):
            base_scores[str(item["feature_name"])] += abs(float(item.get("weight", 0.0)))

    ranked = sorted(base_scores.items(), key=lambda pair: pair[1], reverse=True)[:top_n]
    base_features = [{"feature_name": name, "score": float(score)} for name, score in ranked]

    return {
        "flat_features": [],
        "base_features": base_features,
    }


def build_xai_quality_summary() -> dict[str, Any]:
    jaccards = []
    shap_fidelity_scores = []
    lime_fidelity_scores = []
    shap_stability_scores = []
    lime_stability_scores = []

    selected_ids = select_target_window_ids()

    for window_id in selected_ids:
        explanation = get_or_create_explanation(window_id)

        shap_names = _top_feature_names(explanation.get("shap_weights", []), top_k=5)
        lime_names = _top_feature_names(explanation.get("lime_weights", []), top_k=5)

        jaccard = _jaccard_similarity(shap_names, lime_names)
        if jaccard is not None:
            jaccards.append(jaccard)

        shap_fidelity = _simple_fidelity(explanation.get("shap_weights", []))
        lime_fidelity = _simple_fidelity(explanation.get("lime_weights", []))
        shap_stability = _simple_stability(explanation.get("shap_weights", []))
        lime_stability = _simple_stability(explanation.get("lime_weights", []))

        if shap_fidelity is not None:
            shap_fidelity_scores.append(shap_fidelity)
        if lime_fidelity is not None:
            lime_fidelity_scores.append(lime_fidelity)
        if shap_stability is not None:
            shap_stability_scores.append(shap_stability)
        if lime_stability is not None:
            lime_stability_scores.append(lime_stability)

    return {
        "mean_jaccard_similarity": float(np.mean(jaccards)) if jaccards else None,
        "mean_shap_fidelity": float(np.mean(shap_fidelity_scores)) if shap_fidelity_scores else None,
        "mean_lime_fidelity": float(np.mean(lime_fidelity_scores)) if lime_fidelity_scores else None,
        "mean_shap_stability": float(np.mean(shap_stability_scores)) if shap_stability_scores else None,
        "mean_lime_stability": float(np.mean(lime_stability_scores)) if lime_stability_scores else None,
        "explained_case_count": len(selected_ids),
    }


def build_forensic_outputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if analysis_store.detection_response is None:
        raise InferenceError("No current upload analysis found. Upload a CSV first.")

    quality = analysis_store.xai_quality_summary or build_xai_quality_summary()
    results_by_window = {
        int(row["window_id"]): row for row in analysis_store.detection_response.get("results", [])
    }

    cases = []
    shared_counter = Counter()
    label_counter = Counter()

    for window_id in select_target_window_ids():
        row = results_by_window.get(window_id)
        if row is None:
            continue

        explanation = get_or_create_explanation(window_id)
        shap_names = _top_feature_names(explanation.get("shap_weights", []), top_k=5)
        lime_names = _top_feature_names(explanation.get("lime_weights", []), top_k=5)
        shared = [name for name in shap_names if name in set(lime_names)]

        for name in shared:
            shared_counter[name] += 1

        label_name = str(row.get("predicted_label_name", ""))
        label_counter[label_name] += 1

        jaccard = _jaccard_similarity(shap_names, lime_names)
        shap_fidelity = _simple_fidelity(explanation.get("shap_weights", []))
        lime_fidelity = _simple_fidelity(explanation.get("lime_weights", []))
        shap_stability = _simple_stability(explanation.get("shap_weights", []))
        lime_stability = _simple_stability(explanation.get("lime_weights", []))

        quality_text = (
            f"SHAP/LIME overlap: {len(shared)} shared indicators; "
            f"Jaccard={jaccard:.3f}" if jaccard is not None else "SHAP/LIME overlap unavailable."
        )

        plain_text = (
            f"Window {window_id} was classified as {label_name} because explanation methods "
            f"emphasized indicators such as {', '.join(shared[:3] or shap_names[:3] or lime_names[:3] or ['unknown'])}."
        )

        recommendation = (
            f"Review nearby rows around row {row.get('last_row_id')}, inspect repeated indicators, "
            f"and prioritize this case if the same features recur across multiple windows."
        )

        cases.append(
            {
                "case_id": f"CASE-{window_id:05d}",
                "sample_row_index": int(row.get("last_row_id", -1)),
                "predicted_label_name": label_name,
                "probability_attack": float(row.get("probability_attack", 0.0)),
                "probability_suspicious": float(row.get("probability_suspicious", 0.0)),
                "jaccard_similarity": jaccard,
                "shap_fidelity": shap_fidelity,
                "lime_fidelity": lime_fidelity,
                "shap_stability": shap_stability,
                "lime_stability": lime_stability,
                "explanation_quality_summary": quality_text,
                "plain_language_explanation": plain_text,
                "analyst_recommendation": recommendation,
            }
        )

    cases.sort(
        key=lambda item: (
            0 if item["predicted_label_name"] == "attack" else 1,
            -item["probability_attack"],
            -item["probability_suspicious"],
        )
    )

    summary = {
        "forensic_case_count": len(cases),
        "label_counts": dict(label_counter),
        "top_shared_indicators": [name for name, _ in shared_counter.most_common(10)],
        "mean_jaccard_similarity": quality.get("mean_jaccard_similarity"),
        "mean_shap_fidelity": quality.get("mean_shap_fidelity"),
        "mean_lime_fidelity": quality.get("mean_lime_fidelity"),
        "mean_shap_stability": quality.get("mean_shap_stability"),
        "mean_lime_stability": quality.get("mean_lime_stability"),
    }

    return summary, cases
