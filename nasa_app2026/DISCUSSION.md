# Model discussion

This document records confirmed choices and open implementation questions for Step 2. It is written for a beginner ML developer and can be extended as decisions are made.

## Goal

For each approximately 1 km² equal-area cell with a FIRMS detection today, estimate a probability from 0% to 100% that FIRMS will record at least one hotspot detection in the same cell on the next UTC day.

The probability describes a future satellite detection in FIRMS. It is not a verified probability that a real fire will start: a satellite may miss an existing fire because of clouds, smoke, canopy, or overpass timing.

## Confirmed decisions

| Topic | Decision |
| --- | --- |
| Prediction target | Binary event: at least one FIRMS detection in this cell on the next day (`target_next_day_detected`). |
| Probability output | One probability from 0% to 100% for each eligible cell. Do not predict the number of detections in the first version. |
| Forecast scope | Predict continuation in cells that have a detection today. Predicting newly appearing hotspots in cells with no current detection needs a different training set and reliable negative examples. |
| Input data | Use only the existing prepared FIRMS fields for the first version. Do not add external weather or geographic datasets yet. |
| Model path | Establish a baseline first, then compare a gradient-boosting model. The exact boosting library remains open. |
| Main priority | Best forecast quality. Keep the baseline so the more complex model can be compared against a simple reference. |
| Evaluation | Preserve chronological train/test separation. Report PR-AUC and Brier score, and check probability calibration. |
| Handoff | The trained model and inference interface will be handed to the backend developer, who will build the UI. The UI itself is outside this model discussion. |

## Implementation outline — completed 2026-11-06

The prepared data and baseline were reviewed, the chronological training/validation/calibration/test ranges were fixed, three LightGBM candidates were compared against logistic and constant baselines, the selected model was calibrated, and its callable backend interface and metadata were packaged. Detailed dates, metrics, limitations, and artifact paths follow below.

## Baseline comparison — 2026-11-04

The prepared split was kept intact: official test starts on 2026-08-12. The former training period was divided chronologically again; validation starts on 2026-06-25. A fixed-seed reservoir sample of 150,000 rows per part was used to make this first comparison tractable. Validation selected the logistic configuration; the official test sample was used once afterward.

Three logistic L2 settings were compared. Selection used equal rank weight for PR-AUC (higher is better), Brier score (lower), and 10-bin expected calibration error (lower):

| Validation candidate | L2 | PR-AUC | Brier | ECE | Rank sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strong regularization | 0.01 | 0.4377 | 0.10955 | 0.04724 | 7 |
| **Medium regularization — selected** | **0.001** | **0.4376** | **0.10951** | **0.04721** | **5** |
| Weak regularization | 0.0001 | 0.4376 | 0.10951 | 0.04724 | 6 |

Official test sample comparison:

| Model | PR-AUC | Brier | ECE |
| --- | ---: | ---: | ---: |
| Logistic regression, L2=0.001 | 0.4720 | 0.11644 | 0.03796 |
| Constant historical-rate baseline | 0.1618 | 0.13806 | 0.04223 |

The selected logistic baseline outperformed the constant baseline on all three metrics in this sampled test. These are estimates from 150,000 randomly sampled rows, not full-test metrics. ECE is a coarse calibration check; no post-hoc probability calibration has been applied. The comparison does not yet include gradient boosting. The implementation and function/object notes are in `compare_baselines.py` and `objects.md`; machine-readable metrics are in `processed/baseline_comparison.json`.

## Gradient boosting and packaged model — 2026-11-06

Environment: Python 3.12.14, LightGBM 4.7.0, scikit-learn 1.9.1. Dependencies are recorded in `requirements-model.txt` and were installed only in the project-local `.venv`. The training script uses the 27 explicit columns listed in `compare_baselines.py`; target, target dates, split label, cell identifiers, and coordinates are excluded.

### Fixed chronological split

| Role | Feature dates | Use |
| --- | --- | --- |
| Candidate training | through 2026-06-24 | Train three candidate configurations and the logistic baseline. |
| Configuration validation | 2026-06-25—2026-07-25 | Choose a configuration by equal-weight metric rank sum. |
| Final model fit | 2025-12-31—2026-07-25 | Refit selected configuration on every row before calibration dates (13,215,557 rows). |
| Calibration fit | 2026-07-26—2026-08-05 | Fit sigmoid and isotonic mappings on predictions from the final model; the model did not train on these dates. |
| Calibration selection | 2026-08-06—2026-08-11 | Compare the two mappings; not used to train the base model or fit the calibrators. |
| Official test | from 2026-08-12 | One full evaluation after choices were fixed. No test-based tuning. |

Validation configuration comparison used a reproducible 300,000-row training reservoir and a 150,000-row validation reservoir. The final selected model used the full 13,215,557 permitted training and validation rows. Calibration fit and selection each used 150,000-row reservoirs. The official test was streamed and evaluated in full.

### Candidate results

| Validation model | PR-AUC | Brier | ECE-10 | Rank sum | Fit time on training sample |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant training prevalence | 0.14247 | 0.12261 | 0.02090 | — | — |
| Logistic regression, L2=0.001 | 0.39013 | 0.10937 | 0.03313 | — | 0.41 s |
| LightGBM shallow_regularized | 0.45749 | 0.09901 | 0.02561 | 7 | 4.44 s |
| **LightGBM medium — selected** | **0.46000** | **0.09896** | **0.02520** | **3** | **3.84 s** |
| LightGBM deeper_regularized | 0.45791 | 0.09912 | 0.02772 | 8 | 5.53 s |

Candidate selection used the sum of ranks for PR-AUC (higher is better), Brier (lower), and ECE-10 (lower); ties follow the fixed configuration order. The selected `medium` parameters are 31 leaves, max depth 7, learning rate 0.05, minimum leaf size 200, L2 penalty 2.0, and 250 boosting rounds. The full-data refit took 175.90 seconds.

### Probability calibration

Calibrators were fitted on 150,000 predictions from 2026-07-26—2026-08-05 and compared on separate 150,000 predictions from 2026-08-06—2026-08-11:

| Calibration method | PR-AUC | Brier | ECE-10 |
| --- | ---: | ---: | ---: |
| **Platt sigmoid — selected** | **0.50277** | **0.10128** | **0.00384** |
| Isotonic regression | 0.49498 | 0.10138 | 0.00530 |

Selection primarily minimized Brier score; the rank sum of PR-AUC, Brier, and ECE broke close comparisons. Calibration-set metrics are selection metrics, not a separate unbiased test of the calibrator.

### One-time full official test result

The frozen model and Platt mapping were evaluated on all 5,513,290 official test rows (900,077 positives; observed rate 16.33%):

| PR-AUC | Brier | ECE-10 |
| ---: | ---: | ---: |
| **0.47791** | **0.10922** | **0.01728** |

This is a full-period score, unlike the earlier baseline comparison, whose test metrics used a 150,000-row sample. The earlier logistic test sample scored PR-AUC 0.47203, Brier 0.11644, and ECE-10 0.03796; the constant baseline sample scored 0.16177, 0.13806, and 0.04223. They are useful reference values but not a paired full-test comparison. The full-test calibration bins show some underprediction at low probabilities (mean predicted 5.89% versus observed 7.60% for the 0–10% bin), so probabilities remain estimates subject to temporal drift.

### Artifacts and backend handoff

- `models/firms_next_day_lightgbm.txt` — final LightGBM model.
- `models/calibration.json` — selected sigmoid slope and intercept.
- `models/metadata.json` — feature order, target semantics, split dates, versions, calibration, and input rules.
- `firms_inference.py` — `FirmsNextDayPredictor.predict_one(mapping)` returns probability [0,1], percent, target UTC date, and event description.
- `processed/model_comparison.json` — reproducible report including full official test metrics and calibration bins.

Backend input must contain all 27 prepared features plus `date_utc` in ISO `YYYY-MM-DD` format: `detections`, `mean_brightness`, `mean_scan`, `mean_track`, `mean_bright_t31`, `mean_frp`, `mean_modis_confidence`, `modis_detections`, `viirs_detections`, `archive_detections`, `nrt_detections`, `day_detections`, `night_detections`, `sat_aqua`, `sat_terra`, `sat_n20`, `sat_n21`, `sat_snpp`, `viirs_conf_l`, `viirs_conf_n`, `viirs_conf_h`, `type_0`, `type_1`, `type_2`, `type_3`, `day_of_year_sin`, `day_of_year_cos`. Invalid/missing/non-finite feature values are rejected. Output shape is `{"probability": 0.12, "probability_percent": 12.0, "target_date_utc": "2026-08-13", "event": "at least one FIRMS detection in the same cell"}`. The output is for a cell that already has a FIRMS detection on the source day; it does not predict emergence in cells without current detections. No HTTP route or UI was requested; this package is the ready model and callable inference layer.

## Dataset limits to retain in reports

- The model predicts recorded FIRMS detections, not confirmed fire starts.
- Training rows cover cells with a detection today. This version predicts whether detection persists in that same cell tomorrow; it does not estimate new-fire emergence in other cells.
- A missing FIRMS detection can reflect observation limitations, not necessarily the absence of a fire.
- The prepared data has 45,052,832 cleaned detection records and 20,323,153 cell-day rows with an available next-day target. See `processed/preprocessing_report.json` for full processing statistics.

## Answer history

1. Desired model result: a 0–100% chance of fire occurrence.
2. Clarification: select the first proposed interpretation — probability of a future FIRMS satellite detection in the cell on the next day. Do not label this as confirmed real-fire probability.
3. Model type: baseline first, then gradient boosting.
4. Input data: prepared FIRMS fields only.
5. Positive event: at least one FIRMS detection in the same cell the next day.
6. First-version output: probability only, no expected detection count.
7. Evaluation: chronological split, PR-AUC, Brier score, and calibration.
8. Priority: best prediction quality.
9. Handoff: the backend developer will create the UI; the model work should expose predictions in a backend-friendly form.
10. Baseline comparison: compare the constant historical rate and logistic regression; try three L2 settings and select using validation PR-AUC, Brier score, and ECE ranks. Medium L2 (`0.001`) won; see the results above.
