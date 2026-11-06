# Model handoff for the web application

This guide lists the files to copy into GitHub and shows how a backend can load the model and expose a prediction route. The model must run on the **server**: do not send the model file to a browser or import it from frontend JavaScript. The browser sends prepared features to a backend route; the backend calls the Python predictor and returns JSON.

## 1. Files to upload

Copy these files while preserving the folders:

```text
firms_inference.py
requirements-inference.txt
models/
  firms_next_day_lightgbm.txt
  calibration.json
  metadata.json
```

`firms_inference.py` loads the model feature order from `models/metadata.json`, then applies the saved calibration parameters. Put the `models/` folder beside `firms_inference.py`, as shown above. If the backend stores artifacts elsewhere, pass explicit `model_path`, `calibration_path`, and `metadata_path` when creating `FirmsNextDayPredictor`.

Recommended documentation and audit files:

- `MODEL_HANDOFF.md` — this deployment and upload guide.
- `README.md` — project overview and model summary.
- `DISCUSSION.md` — decisions, split dates, metrics, and limitations.
- `processed/model_comparison.json` — machine-readable model results, including the full official test evaluation.

Upload these only if the backend developer will reproduce or retrain the model:

- `train_firms_model.py`
- `requirements-model.txt`
- `objects.md`
- `processed/baseline_comparison.json`

The application does **not** need the large training data, source CSV/ZIP archives, `.venv/`, Python cache folders, or preprocessing scripts. The prepared daily table is needed only when making predictions from raw FIRMS records; the model itself accepts the 27 already prepared features.

## 2. Backend installation and model loading

In the Python backend environment, install the small inference dependency set:

```bash
python -m pip install -r requirements-inference.txt
```

Load the model once when the backend process starts, then reuse that object for requests. Do not reload the 0.9 MB model on every request.

```python
from firms_inference import FirmsNextDayPredictor

predictor = FirmsNextDayPredictor()
```

The project-local model uses LightGBM 4.7.0 and NumPy 2.5.4. The inference package does not need scikit-learn or the training script.

## 3. Prediction input

The backend must send every feature below, using the same definitions and units as the prepared table. The source date is `date_utc` in UTC as `YYYY-MM-DD`.

```text
detections, mean_brightness, mean_scan, mean_track, mean_bright_t31,
mean_frp, mean_modis_confidence, modis_detections, viirs_detections,
archive_detections, nrt_detections, day_detections, night_detections,
sat_aqua, sat_terra, sat_n20, sat_n21, sat_snpp, viirs_conf_l, viirs_conf_n,
viirs_conf_h, type_0, type_1, type_2, type_3, day_of_year_sin, day_of_year_cos
```

These fields are the prepared FIRMS cell-day features. The prediction target is whether at least one FIRMS detection is recorded in that same cell on the next UTC day. This model only applies to cells that already had a detection on the input day. A missing or non-numeric feature, NaN, infinity, or invalid date is rejected; the backend should return a client error (HTTP 422 or 400) and should not silently replace it with zero.

## 4. Example backend route

The following FastAPI example shows the boundary between the web application and the predictor. Add FastAPI and an ASGI server such as Uvicorn to the backend's own dependency list; they are not model dependencies in `requirements-inference.txt`.

Save this snippet as `model_api.py` in the backend service. After installing that service's FastAPI/Uvicorn dependencies, start it with `uvicorn model_api:app --host 0.0.0.0 --port 8000` (use the deployment platform's normal process command in production).

```python
from fastapi import FastAPI, HTTPException
from firms_inference import FirmsNextDayPredictor

app = FastAPI()
predictor = FirmsNextDayPredictor()  # loaded once at startup


@app.post("/api/predict")
def predict(payload: dict):
    try:
        return predictor.predict_one(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
```

Illustrative request body with all 27 required features:

```json
{
  "date_utc": "2026-08-12",
  "detections": 3,
  "mean_brightness": 340.2,
  "mean_scan": 0.45,
  "mean_track": 0.48,
  "mean_bright_t31": 298.1,
  "mean_frp": 12.4,
  "mean_modis_confidence": 0,
  "modis_detections": 0,
  "viirs_detections": 3,
  "archive_detections": 0,
  "nrt_detections": 3,
  "day_detections": 3,
  "night_detections": 0,
  "sat_aqua": 0,
  "sat_terra": 0,
  "sat_n20": 3,
  "sat_n21": 0,
  "sat_snpp": 0,
  "viirs_conf_l": 0,
  "viirs_conf_n": 3,
  "viirs_conf_h": 0,
  "type_0": 0,
  "type_1": 0,
  "type_2": 0,
  "type_3": 0,
  "day_of_year_sin": 0.7,
  "day_of_year_cos": -0.7
}
```

Example response:

```json
{
  "probability": 0.12,
  "probability_percent": 12.0,
  "target_date_utc": "2026-08-13",
  "event": "at least one FIRMS detection in the same cell"
}
```

Frontend JavaScript calls the backend route; it does not load the LightGBM artifact:

```javascript
const response = await fetch("/api/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(preparedCellDay)
});

if (!response.ok) throw new Error("Prediction request failed");
const prediction = await response.json();
// Display prediction.probability_percent and prediction.target_date_utc.
```

If frontend and backend use different origins, configure CORS on the backend for the web app's exact origin. Do not expose an unrestricted prediction endpoint publicly without the application's normal authentication and rate limits.

## 5. Output meaning and model limits

- `probability` is the calibrated value in [0, 1]; `probability_percent` is the same value multiplied by 100.
- `target_date_utc` is one calendar day after `date_utc`.
- This is a probability of a future FIRMS satellite detection, not proof of a fire or a probability that a fire starts.
- The input row must describe a cell-day with at least one detection today. This model does not predict new hotspots in cells without a current detection.
- The model was evaluated on a full historical test interval. Future data and satellite coverage can differ, so monitor calibration after deployment.
- The trained model is frozen. Re-training or changing feature calculations requires re-evaluation and replacing the model, calibration, and metadata files together.

For the exact split dates, metrics, calibration comparison, feature order, and training history, see `DISCUSSION.md`, `models/metadata.json`, and `processed/model_comparison.json`.
