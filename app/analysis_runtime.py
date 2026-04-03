from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CurrentAnalysisStore:
    filename: str | None = None
    detection_response: dict[str, Any] | None = None
    global_summary: dict[str, Any] | None = None
    xai_quality_summary: dict[str, Any] | None = None
    forensic_summary: dict[str, Any] | None = None
    forensic_cases: list[dict[str, Any]] = field(default_factory=list)
    explanation_cache: dict[int, dict[str, Any]] = field(default_factory=dict)

    def clear(self) -> None:
        self.filename = None
        self.detection_response = None
        self.global_summary = None
        self.xai_quality_summary = None
        self.forensic_summary = None
        self.forensic_cases = []
        self.explanation_cache = {}


analysis_store = CurrentAnalysisStore()
