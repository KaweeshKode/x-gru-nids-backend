import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf


class InferenceError(Exception):
    """Raised when uploaded data cannot be processed for inference."""


@dataclass
class PredictionContext:
    sequence_array: np.ndarray
    predicted_probabilities: np.ndarray
    predicted_label_ids: np.ndarray
    sequence_metadata: pd.DataFrame


class MLService:
    CATEGORICAL_FEATURE_COLUMNS = ["proto", "state", "service"]
    CLASS_NAMES = ["normal", "suspicious", "attack"]
    PRIMARY_TIME_COLUMN = "ltime"
    SECONDARY_TIME_COLUMN = "stime"

    def __init__(self) -> None:
        self.backend_root = Path(__file__).resolve().parent.parent
        self.artifacts_dir = self.backend_root / "artifacts"
        self.model_path = self.artifacts_dir / "model" / "cnn_gru_intrusion_model.keras"
        self.preprocessing_dir = self.artifacts_dir / "preprocessing"
        self.sequence_dir = self.artifacts_dir / "sequences"

        self.scaler = None
        self.category_mappings: dict[str, dict[str, int]] = {}
        self.scaler_columns_from_file: list[str] = []
        self.scaler_columns: list[str] = []
        self.feature_columns: list[str] = []
        self.sequence_length = 10
        self.sequence_stride = 1
        self.class_names = list(self.CLASS_NAMES)
        self.model = None
        self.current_context: Optional[PredictionContext] = None
        self.is_ready = False

        self._load_artifacts()

    def _load_json(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _extract_string_list(payload: Any, preferred_keys: list[str] | None = None) -> list[str]:
        if isinstance(payload, list):
            return [str(x) for x in payload]

        if isinstance(payload, dict):
            for key in preferred_keys or []:
                value = payload.get(key)
                if isinstance(value, list):
                    return [str(x) for x in value]

            for value in payload.values():
                if isinstance(value, list):
                    return [str(x) for x in value]

        return []

    def _resolve_scaler_columns(self) -> list[str]:
        if self.scaler is not None and hasattr(self.scaler, "feature_names_in_"):
            try:
                return [str(x) for x in self.scaler.feature_names_in_.tolist()]
            except Exception:
                return [str(x) for x in self.scaler.feature_names_in_]

        return list(self.scaler_columns_from_file)

    def _load_artifacts(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. "
                "Copy the trained artifacts into backend/artifacts before starting the API."
            )

        scaler_path = self.preprocessing_dir / "scaler.pkl"
        category_mapping_path = self.preprocessing_dir / "category_mappings.json"
        scaler_columns_path = self.preprocessing_dir / "scaler_columns.json"
        sequence_features_path = self.sequence_dir / "sequence_feature_columns.json"
        sequence_config_path = self.sequence_dir / "sequence_build_config.json"

        with open(scaler_path, "rb") as handle:
            self.scaler = pickle.load(handle)

        self.category_mappings = self._load_json(category_mapping_path)

        scaler_column_payload = self._load_json(scaler_columns_path)
        self.scaler_columns_from_file = self._extract_string_list(
            scaler_column_payload,
            ["scaled_numeric_feature_columns", "scaler_columns", "columns"],
        )
        self.scaler_columns = self._resolve_scaler_columns()

        sequence_feature_payload = self._load_json(sequence_features_path)
        self.feature_columns = self._extract_string_list(
            sequence_feature_payload,
            ["feature_columns", "columns"],
        )

        sequence_config = self._load_json(sequence_config_path)
        self.sequence_length = int(sequence_config.get("sequence_length", sequence_config.get("window_size", 10)))
        self.sequence_stride = int(sequence_config.get("sequence_stride", sequence_config.get("stride", 1)))

        self.model = tf.keras.models.load_model(self.model_path)
        self.is_ready = True

    @staticmethod
    def normalize_column_name(name: str) -> str:
        return str(name).strip().lower().replace(" ", "")

    @staticmethod
    def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        safe_denominator = denominator.replace(0, np.nan)
        result = numerator / safe_denominator
        return result.replace([np.inf, -np.inf], 0).fillna(0)

    def clean_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        cleaned_dataset = dataset.copy()
        cleaned_dataset.columns = [self.normalize_column_name(column) for column in cleaned_dataset.columns]
        cleaned_dataset = cleaned_dataset.replace([np.inf, -np.inf], np.nan)

        if "attack_cat" in cleaned_dataset.columns:
            cleaned_dataset["attack_cat"] = (
                cleaned_dataset["attack_cat"].fillna("normal").astype(str).str.strip().str.lower()
            )
        if "label" in cleaned_dataset.columns:
            cleaned_dataset["label"] = pd.to_numeric(cleaned_dataset["label"], errors="coerce").fillna(0).astype(int)

        skip_numeric_cast = {"srcip", "dstip", "proto", "state", "service", "attack_cat", "source_file"}
        candidate_numeric_columns = [column for column in cleaned_dataset.columns if column not in skip_numeric_cast]

        for column in candidate_numeric_columns:
            cleaned_dataset[column] = pd.to_numeric(cleaned_dataset[column], errors="coerce")

        for column in self.CATEGORICAL_FEATURE_COLUMNS:
            if column in cleaned_dataset.columns:
                cleaned_dataset[column] = cleaned_dataset[column].astype(str)

        numeric_columns = cleaned_dataset.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            cleaned_dataset[numeric_columns] = cleaned_dataset[numeric_columns].fillna(0)

        cleaned_dataset = cleaned_dataset.replace([np.inf, -np.inf], 0)
        return cleaned_dataset

    def engineer_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        engineered_dataset = dataset.copy()

        if {"sbytes", "dbytes"}.issubset(engineered_dataset.columns):
            engineered_dataset["bytes_total"] = engineered_dataset["sbytes"] + engineered_dataset["dbytes"]
            engineered_dataset["byte_ratio"] = self.safe_divide(
                engineered_dataset["sbytes"], engineered_dataset["dbytes"] + 1
            )

        if {"spkts", "dpkts"}.issubset(engineered_dataset.columns):
            engineered_dataset["pkts_total"] = engineered_dataset["spkts"] + engineered_dataset["dpkts"]
            engineered_dataset["pkt_ratio"] = self.safe_divide(
                engineered_dataset["spkts"], engineered_dataset["dpkts"] + 1
            )

        if {"sttl", "dttl"}.issubset(engineered_dataset.columns):
            engineered_dataset["ttl_gap"] = (engineered_dataset["sttl"] - engineered_dataset["dttl"]).abs()

        if {"sload", "dload"}.issubset(engineered_dataset.columns):
            engineered_dataset["load_total"] = engineered_dataset["sload"] + engineered_dataset["dload"]

        numeric_columns = engineered_dataset.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            engineered_dataset[numeric_columns] = engineered_dataset[numeric_columns].fillna(0)

        return engineered_dataset

    def _sort_dataset_by_time(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if self.PRIMARY_TIME_COLUMN in dataset.columns and self.SECONDARY_TIME_COLUMN in dataset.columns:
            return dataset.sort_values(
                by=[self.PRIMARY_TIME_COLUMN, self.SECONDARY_TIME_COLUMN],
                kind="mergesort",
            ).reset_index(drop=True)
        return dataset.reset_index(drop=True)

    def _encode_categorical_features(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        encoded = feature_table.copy()
        for column, mapping in self.category_mappings.items():
            if column in encoded.columns:
                encoded[column] = encoded[column].astype(str).map(mapping).fillna(-1).astype(int)
        return encoded

    def _coerce_numeric_columns(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        coerced = frame.copy()
        for column in columns:
            if column in coerced.columns:
                coerced[column] = pd.to_numeric(coerced[column], errors="coerce").fillna(0)
        return coerced

    def _prepare_feature_frame(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        prepared = self.clean_dataset(dataset)
        prepared = self.engineer_features(prepared)
        prepared = self._sort_dataset_by_time(prepared).reset_index(drop=True)

        if "row_id" not in prepared.columns:
            prepared["row_id"] = np.arange(len(prepared))

        metadata = pd.DataFrame(
            {
                "row_id": prepared["row_id"].astype(int),
                "ltime": prepared["ltime"] if "ltime" in prepared.columns else np.arange(len(prepared)),
                "stime": prepared["stime"] if "stime" in prepared.columns else np.arange(len(prepared)),
            }
        ).reset_index(drop=True)

        feature_table = pd.DataFrame(index=prepared.index)

        for column in set(self.feature_columns) | set(self.scaler_columns) | set(self.category_mappings.keys()):
            if column in prepared.columns:
                feature_table[column] = prepared[column]
            elif column in self.CATEGORICAL_FEATURE_COLUMNS:
                feature_table[column] = "missing"
            else:
                feature_table[column] = 0

        # Encode categoricals BEFORE scaling because the fitted scaler expects encoded proto/service/state too.
        feature_table = self._encode_categorical_features(feature_table)

        # Ensure exact scaler input order and numeric type.
        for column in self.scaler_columns:
            if column not in feature_table.columns:
                feature_table[column] = 0
        feature_table = self._coerce_numeric_columns(feature_table, self.scaler_columns)
        feature_table[self.scaler_columns] = self.scaler.transform(feature_table[self.scaler_columns])

        # Ensure final model feature columns exist and are numeric.
        for column in self.feature_columns:
            if column not in feature_table.columns:
                feature_table[column] = 0
        feature_table = self._coerce_numeric_columns(feature_table, self.feature_columns)
        feature_table[self.feature_columns] = feature_table[self.feature_columns].fillna(0)

        return feature_table, metadata

    def create_sequences(self, feature_table: pd.DataFrame, metadata: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        if len(feature_table) < self.sequence_length:
            raise InferenceError(
                f"Uploaded CSV has only {len(feature_table)} rows. "
                f"At least {self.sequence_length} rows are required to build one sequence window."
            )

        feature_values = feature_table[self.feature_columns].values.astype(np.float32)
        sequence_rows = []
        sequence_metadata_rows = []

        for start_index in range(0, len(feature_table) - self.sequence_length + 1, self.sequence_stride):
            end_index = start_index + self.sequence_length
            sequence_rows.append(feature_values[start_index:end_index])
            sequence_metadata_rows.append(
                {
                    "last_row_id": int(metadata.iloc[end_index - 1]["row_id"]),
                    "last_ltime": float(metadata.iloc[end_index - 1]["ltime"]),
                    "last_stime": float(metadata.iloc[end_index - 1]["stime"]),
                }
            )

        return np.asarray(sequence_rows, dtype=np.float32), pd.DataFrame(sequence_metadata_rows)

    def predict_traffic(self, dataframe: pd.DataFrame, filename: str) -> dict[str, Any]:
        if dataframe.empty:
            raise InferenceError("The uploaded CSV is empty.")

        feature_table, metadata = self._prepare_feature_frame(dataframe)
        sequence_array, sequence_metadata = self.create_sequences(feature_table, metadata)

        predicted_probabilities = self.model.predict(sequence_array, verbose=0)
        predicted_label_ids = np.argmax(predicted_probabilities, axis=1)

        results = []
        label_counts = {label_name: 0 for label_name in self.class_names}

        for index, label_id in enumerate(predicted_label_ids):
            label_name = self.class_names[int(label_id)]
            label_counts[label_name] += 1
            probabilities = predicted_probabilities[index]

            results.append(
                {
                    "window_id": int(index),
                    "last_row_id": int(sequence_metadata.iloc[index]["last_row_id"]),
                    "predicted_label_id": int(label_id),
                    "predicted_label_name": label_name,
                    "probability_normal": float(probabilities[0]),
                    "probability_suspicious": float(probabilities[1]),
                    "probability_attack": float(probabilities[2]),
                    "alert_score": float(probabilities[1] + probabilities[2]),
                }
            )

        self.current_context = PredictionContext(
            sequence_array=sequence_array,
            predicted_probabilities=predicted_probabilities,
            predicted_label_ids=predicted_label_ids,
            sequence_metadata=sequence_metadata,
        )

        return {
            "filename": filename,
            "summary": {
                "total_rows": int(len(dataframe)),
                "total_windows": int(len(results)),
                "suspicious_or_attack_windows": int(label_counts["suspicious"] + label_counts["attack"]),
                "label_counts": {key: int(value) for key, value in label_counts.items()},
            },
            "results": results,
        }

    def get_background_sequences(self, max_samples: int = 50) -> Optional[np.ndarray]:
        if self.current_context is None or len(self.current_context.sequence_array) == 0:
            return None
        return self.current_context.sequence_array[: min(max_samples, len(self.current_context.sequence_array))]

    def get_sequence_for_explanation(self, window_id: int) -> tuple[np.ndarray, int, str]:
        if self.current_context is None:
            raise InferenceError("No uploaded traffic is available yet. Upload a CSV first.")
        if window_id < 0 or window_id >= len(self.current_context.sequence_array):
            raise InferenceError(f"Window {window_id} does not exist in the current upload.")

        predicted_label_id = int(self.current_context.predicted_label_ids[window_id])
        predicted_label_name = self.class_names[predicted_label_id]
        return self.current_context.sequence_array[window_id], predicted_label_id, predicted_label_name


ml_engine = MLService()
