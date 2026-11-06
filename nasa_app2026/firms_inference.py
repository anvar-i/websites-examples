"""Small backend-facing loader for calibrated next-day FIRMS predictions."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "firms_next_day_lightgbm.txt"
CALIBRATION_PATH = ROOT / "models" / "calibration.json"
METADATA_PATH = ROOT / "models" / "metadata.json"


def _apply_calibrator(raw_score: np.ndarray, calibrator: dict) -> np.ndarray:
    """Map raw LightGBM scores to probabilities using packaged JSON parameters."""
    if calibrator["method"] == "platt_sigmoid":
        z = np.clip(float(calibrator["slope"]) * raw_score + float(calibrator["intercept"]), -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))
    if calibrator["method"] == "isotonic":
        return np.interp(raw_score, np.asarray(calibrator["x_thresholds"]), np.asarray(calibrator["y_thresholds"]))
    raise ValueError(f"Unsupported calibration method: {calibrator['method']}")


class FirmsNextDayPredictor:
    """Load the packaged model and return a calibrated next-UTC-day probability."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        calibration_path: Path = CALIBRATION_PATH,
        metadata_path: Path = METADATA_PATH,
    ):
        """Load the model, feature order, and JSON calibration parameters once."""
        self.model = lgb.Booster(model_file=str(model_path))
        self.calibrator = json.loads(calibration_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.feature_columns = metadata["feature_columns"]
        model_features = self.model.feature_name()
        if model_features and model_features != self.feature_columns:
            raise ValueError("Metadata feature order does not match the packaged model")

    def predict_one(self, features: Mapping[str, object]) -> dict:
        """Predict one eligible cell-day; require every feature and an ISO UTC date."""
        if "date_utc" not in features:
            raise ValueError("date_utc is required to determine the next UTC target date")
        try:
            values = [float(features[column]) for column in self.feature_columns]
        except KeyError as exc:
            raise ValueError(f"Missing required model feature: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Model features must be numeric: {exc}") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Feature row contains NaN or infinite values")
        vector = np.asarray([values], dtype=np.float32)
        date_text = str(features["date_utc"])
        source_date = date.fromisoformat(date_text)
        if source_date.isoformat() != date_text:
            raise ValueError("date_utc must use the ISO YYYY-MM-DD format")
        raw_score = self.model.predict(vector, raw_score=True)
        probability = float(_apply_calibrator(raw_score, self.calibrator)[0])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("Model returned a probability outside [0, 1]")
        return {
            "probability": probability,
            "probability_percent": probability * 100.0,
            "target_date_utc": (source_date + timedelta(days=1)).isoformat(),
            "event": "at least one FIRMS detection in the same cell",
        }
