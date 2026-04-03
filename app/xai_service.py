import re

import lime.lime_tabular
import numpy as np
import shap


class XAIService:
    def __init__(self) -> None:
        self.model = None
        self.feature_columns: list[str] = []
        self.sequence_length: int = 10
        self.class_names: list[str] = ["normal", "suspicious", "attack"]
        self.flat_feature_names: list[str] = []
        self.lime_explainer = None
        self.shap_explainer = None
        self.is_ready = False

    def prepare(
        self,
        model,
        background_sequences: np.ndarray,
        feature_columns: list[str],
        sequence_length: int,
        class_names: list[str],
    ) -> None:
        self.model = model
        self.feature_columns = list(feature_columns)
        self.sequence_length = int(sequence_length)
        self.class_names = list(class_names)
        self.flat_feature_names = [
            f"t{time_index}_{feature_name}"
            for time_index in range(self.sequence_length)
            for feature_name in self.feature_columns
        ]

        background_flat = background_sequences.reshape(background_sequences.shape[0], -1)

        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=background_flat,
            feature_names=self.flat_feature_names,
            class_names=self.class_names,
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

        shap_background = shap.kmeans(background_flat, min(10, len(background_flat)))
        self.shap_explainer = shap.KernelExplainer(self._predict_wrapper, shap_background)
        self.is_ready = True

    def _predict_wrapper(self, flat_input: np.ndarray) -> np.ndarray:
        if len(flat_input.shape) == 1:
            flat_input = flat_input.reshape(1, -1)

        reshaped_input = flat_input.reshape((-1, self.sequence_length, len(self.feature_columns)))
        return self.model.predict(reshaped_input, verbose=0)

    @staticmethod
    def _extract_flat_feature_name(condition_or_feature: str) -> str:
        match = re.search(r"(t\d+_[A-Za-z0-9_]+)", str(condition_or_feature))
        return match.group(1) if match else str(condition_or_feature).strip()

    @staticmethod
    def _base_feature_name(flat_feature_name: str) -> str:
        if flat_feature_name.startswith("t") and "_" in flat_feature_name:
            return flat_feature_name.split("_", 1)[1]
        return flat_feature_name

    @staticmethod
    def _sort_signed_items(items: dict[str, float], top_k: int = 10) -> list[dict]:
        sorted_items = sorted(items.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
        return [{"feature_name": name, "weight": float(weight)} for name, weight in sorted_items]

    def _aggregate_lime(self, explanation, predicted_label_id: int) -> list[dict]:
        weights_by_base_feature: dict[str, float] = {}

        for condition_or_feature, weight in explanation.as_list(label=predicted_label_id):
            flat_feature_name = self._extract_flat_feature_name(condition_or_feature)
            base_feature_name = self._base_feature_name(flat_feature_name)
            weights_by_base_feature[base_feature_name] = (
                weights_by_base_feature.get(base_feature_name, 0.0) + float(weight)
            )

        return self._sort_signed_items(weights_by_base_feature, top_k=10)

    def _aggregate_shap(self, shap_values: np.ndarray) -> list[dict]:
        reshaped_values = shap_values.reshape(self.sequence_length, len(self.feature_columns))

        # keep sign while aggregating across time
        weights_by_base_feature = reshaped_values.sum(axis=0)

        signed_items = {
            self.feature_columns[index]: float(weights_by_base_feature[index])
            for index in range(len(self.feature_columns))
        }
        return self._sort_signed_items(signed_items, top_k=10)

    def explain_instance(
        self,
        instance_3d: np.ndarray,
        predicted_label_id: int,
        predicted_label_name: str,
        window_id: int,
    ) -> dict:
        if not self.is_ready:
            raise ValueError("XAI engine is not ready. Upload data first.")

        instance_flat = instance_3d.reshape(-1)

        lime_explanation = self.lime_explainer.explain_instance(
            data_row=instance_flat,
            predict_fn=self._predict_wrapper,
            num_features=min(60, len(self.flat_feature_names)),
            top_labels=len(self.class_names),
            num_samples=500,
        )
        lime_weights = self._aggregate_lime(lime_explanation, predicted_label_id)

        shap_values = self.shap_explainer.shap_values(instance_flat.reshape(1, -1), nsamples=100)
        if isinstance(shap_values, list):
            shap_for_class = np.asarray(shap_values[predicted_label_id]).reshape(-1)
        else:
            shap_array = np.asarray(shap_values)
            if shap_array.ndim == 3:
                shap_for_class = shap_array[0, :, predicted_label_id].reshape(-1)
            else:
                shap_for_class = shap_array.reshape(-1)

        shap_weights = self._aggregate_shap(shap_for_class)

        return {
            "window_id": int(window_id),
            "predicted_label_name": predicted_label_name,
            "lime_weights": lime_weights,
            "shap_weights": shap_weights,
        }


xai_engine = XAIService()