# Объекты и функции

Назначение: заметки к объектам файла `compare_baselines.py`, который сравнивает историческую долю детекций и три настройки логистической регрессии.

## Конфигурация

- `DATA_PATH` — путь к подготовленному CSV в gzip.
- `OUTPUT_PATH` — JSON с параметрами, метриками и результатом выбора.
- `VALIDATION_START` — начало временной validation-части внутри прежнего train (`2026-06-25`). Даты после неё и до official test используются для выбора настройки.
- `MAX_SAMPLE_ROWS` — верхняя граница случайной reservoir-выборки каждой части. Случайное зерно обеспечивает повторяемый результат.
- `BATCH_SIZE`, `EPOCHS` — размер мини-пакета и число проходов обучения логистической регрессии.
- `FEATURE_COLUMNS` — подготовленные FIRMS-признаки; даты, координаты, цель и индикатор split исключены.
- `LogisticConfig` — неизменяемая пара «название настройки, сила L2-регуляризации».
- `CONFIGS` — три проверяемые силы регуляризации: сильная (`0.01`), средняя (`0.001`), слабая (`0.0001`).

## Функции

- `collect_samples()` — одним проходом читает gzip, считает баланс классов, выделяет chronological train/validation/test и собирает воспроизводимые reservoir-выборки.
- `scale_features()` — стандартизирует семь непрерывных признаков по sampled train; те же параметры применяет к validation/test.
- `sigmoid()` — переводит логиты модели в вероятности.
- `fit_logistic()` — обучает логистическую регрессию мини-пакетным SGD с заданной L2-регуляризацией.
- `score_probabilities()` — рассчитывает PR-AUC (average precision), Brier score и десятиинтервальную ECE-калибровку.
- `main()` — сравнивает постоянную историческую вероятность с тремя конфигурациями, выбирает лучшую по сумме рангов метрик на validation, затем один раз оценивает её и constant baseline на official test и сохраняет JSON.

## Ограничения исходного baseline-скрипта

- Для скорости сравнение и выбор конфигурации выполняются на случайных выборках до `MAX_SAMPLE_ROWS` строк на часть; метрики поэтому являются оценочными, не полным проходом по каждому ряду.
- Сумма рангов даёт равный вес PR-AUC, Brier score и ECE; это явное рабочее правило выбора, а не ранее подтверждённое продуктовое требование.
- Описание выше относится только к `compare_baselines.py`; результаты бустинга и калибровки зафиксированы ниже в описании `train_firms_model.py`.

## Объекты `train_firms_model.py`

Назначение: потоково прочитать готовую таблицу, сравнить заранее выбранные конфигурации LightGBM, обучить финальную модель на всех разрешённых строках, выбрать калибратор на будущих по времени выборках, один раз оценить модель на полном official test и сохранить артефакты.

### Константы и конфигурации

- `DATA_PATH`, `MODEL_DIR`, `REPORT_PATH`, `MODEL_PATH`, `METADATA_PATH`, `CALIBRATION_PATH` — входной набор и пути выходных отчётов/артефактов.
- `FEATURE_COLUMNS`, `TARGET_COLUMN` — точный порядок 27 входных признаков и целевая бинарная метка.
- `TRAIN_END`, `VALIDATION_START`, `VALIDATION_END`, `CALIBRATION_FIT_START`, `CALIBRATION_FIT_END`, `CALIBRATION_SELECT_START`, `CALIBRATION_SELECT_END`, `OFFICIAL_TEST_START` — фиксированные временные границы без случайного перемешивания дат.
- `MAX_TRAIN_SAMPLE`, `MAX_EVAL_SAMPLE`, `READ_CHUNK_ROWS`, `SEED` — размеры воспроизводимых выборок, потокового блока и случайное зерно.
- `BoostConfig` — неизменяемое описание сложности дерева, learning rate, регуляризации и числа раундов.
- `CONFIGS` — три фиксированных варианта: `shallow_regularized`, `medium`, `deeper_regularized`.

### Функции

- `assign_period()` — назначает строке один временной этап по `date_utc` и `time_split`.
- `row_to_features()` — извлекает входы в установленном порядке и отклоняет пропуски, нечисловые значения, NaN и бесконечности.
- `collect_training_and_samples()` — потоково читает gzip; собирает полный массив разрешённых train+validation строк для финального обучения и выборки для сравнения/калибровки.
- `score_probabilities()` — считает average precision/PR-AUC, Brier, десятиинтервальную ECE и калибровочные bins.
- `metric_rank_sum()` — суммирует ранги validation-кандидатов по заранее заданным метрикам.
- `lightgbm_params()` — задаёт детерминированные параметры CPU-обучения для конфигурации.
- `fit_booster()` — создаёт LightGBM Dataset, обучает модель и возвращает время обучения.
- `fit_platt()` — обучает sigmoid/Platt mapping на raw scores отдельного calibration-fit набора.
- `apply_calibrator()` — применяет сохранённые параметры Platt или пороги isotonic.
- `select_calibrator()` — сравнивает калибраторы на отдельном, более позднем calibration-selection наборе.
- `score_official_test()` — потоково рассчитывает метрики на всех официальных test-строках.
- `main()` — связывает сбор данных, baseline и бустинг-кандидатов, ранжирование, финальный full-data fit, калибровку, полный test и сохранение JSON/модели.

## Объекты `firms_inference.py`

- `ROOT`, `MODEL_PATH`, `CALIBRATION_PATH` — пути к пакетируемым артефактам относительно файла модуля.
- `METADATA_PATH` — путь к metadata JSON, из которого при запуске читается верный порядок признаков.
- `_apply_calibrator()` — применяет сохранённый Platt sigmoid или isotonic mapping без scikit-learn.
- `FirmsNextDayPredictor` — загружает LightGBM text model, metadata и JSON-калибратор один раз при создании.
- `FirmsNextDayPredictor.predict_one(features)` — проверяет `date_utc` и полный набор признаков, возвращает calibrated probability [0,1], проценты, дату цели UTC и описание события.

## Ресурсы среды

- `requirements-model.txt` фиксирует LightGBM, scikit-learn и NumPy версии, нужные для обучения и сравнения.
- `requirements-inference.txt` содержит только LightGBM и NumPy для runtime без scikit-learn.
- `MODEL_HANDOFF.md` — ручной GitHub upload checklist и web/backend интеграционная инструкция.
- `models/firms_next_day_lightgbm.txt` — обученные деревья; `models/calibration.json` — параметры Platt; `models/metadata.json` — контракт и происхождение.
- `processed/model_comparison.json` — конфигурации, метрики на временных этапах, времена обучения, полная оценка official test и калибровочные bins.
