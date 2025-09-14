# applied_sklearn_for_ml — IsolationForest Anomaly Detection

Play with different scikit‑learn algorithms. This repo includes a small, production‑style setup for **unsupervised anomaly detection** using **IsolationForest** with a clean train/inference split.

---

## Why this template?
- **Reproducible**: all hyper‑params live in a YAML config.
- **Safe preproc**: imputation + scaling wrapped in a `Pipeline` / `ColumnTransformer`.
- **Deployable**: train once, **save the fitted pipeline**, reuse for inference.
- **Traceable**: metadata (features, sklearn version, etc.) saved alongside the model.

---

## Project layout
```
anomaly_iforest/
├─ src/
│  ├─ pipelines.py           # build_pipeline() & preprocessing
│  ├─ utils.py               # config I/O, validation, annotation helpers
├─ scripts/
│  ├─ train.py               # fit & save the pipeline
│  ├─ inference.py           # load, score, export, optional Plotly viz
├─ configs/
│  └─ config.yaml            # features & hyperparameters
├─ models/
│  ├─ iforest_pipeline.joblib    (generated)
│  └─ iforest_meta.json          (generated)
├─ data/
│  ├─ train.csv
│  └─ new_data.csv
├─ requirements.txt
└─ README.md
```

---

## Installation
```bash
python -m venv .venv && source .venv/bin/activate   # or: conda create -n iforest python=3.11
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
scikit-learn
joblib
plotly
pyyaml
```

---

## Data expectations
- A tabular CSV with **numeric features**. By default we use two petrophysics features:
  - `NPHI`, `RHOB`
- No target column is required (unsupervised). Missing values are handled by a median **imputer**.

**Example (data/train.csv)**
```csv
NPHI,RHOB
0.18,2.45
0.22,2.50
0.12,2.60
...
```

> If your CSV has extra columns, it’s fine. We’ll **select** the ones listed in `configs/config.yaml`.

---

## Configuration
`configs/config.yaml` drives everything.

```yaml
features: [NPHI, RHOB]
contamination: 0.10           # expected anomaly rate (affects threshold)
random_state: 42
imputer_strategy: median      # median | mean | most_frequent
scaler: standard              # standard | minmax | robust
model_path: models/iforest_pipeline.joblib
meta_path: models/iforest_meta.json
```

- **contamination**: rough guess of the anomaly proportion. Used by IsolationForest to set its internal decision threshold. If you plan to pick a **custom threshold**, you can still set a value here but you’ll override later (see below).
- **scaler**: choose from `standard`, `minmax`, or `robust`.

---

## Train
```bash
python scripts/train.py --data data/train.csv --config configs/config.yaml
```
Outputs:
- `models/iforest_pipeline.joblib` — the **fitted** preproc + IsolationForest.
- `models/iforest_meta.json` — metadata (features, sklearn version, etc.).

---

## Inference / Scoring
```bash
python scripts/inference.py --data data/new_data.csv --config configs/config.yaml --out scored.csv --plot
```
Outputs:
- `scored.csv` — original data plus columns:
  - `anomaly_score` — **higher = more normal**, lower = more anomalous.
  - `anomaly_pred`  — `1` = inlier, `-1` = outlier (sklearn convention).
  - `anomaly_flag`  — boolean convenience column (`True` for anomalies).
- If `--plot` is passed, an interactive Plotly **scatter** appears (melted view with both features, colored by variable, symbol by anomaly).

---

## Using as a library
```python
from joblib import load
import pandas as pd

pipe = load("models/iforest_pipeline.joblib")
df = pd.read_csv("data/new_data.csv")
scores = pipe.decision_function(df[["NPHI","RHOB"]])
preds  = pipe.predict(df[["NPHI","RHOB"]])  # 1=inlier, -1=outlier
```

---

## Custom decision threshold (optional)
Sometimes you’ll want to define the anomaly cut‑off yourself (e.g., bottom 5% of scores):

```python
import numpy as np
from joblib import load
import pandas as pd

pipe = load("models/iforest_pipeline.joblib")
train = pd.read_csv("data/train.csv")
train_scores = pipe.decision_function(train[["NPHI","RHOB"]])
custom_thr = np.quantile(train_scores, 0.05)  # bottom 5% anomalous

new = pd.read_csv("data/new_data.csv")
scores = pipe.decision_function(new[["NPHI","RHOB"]])
custom_pred = (scores < custom_thr).astype(int)  # 1=anomaly, 0=normal
```

> If you adopt a custom threshold, **store it** (e.g., in `meta.json`) so inference is reproducible.

---

## Extending the template
- **More features**: add to `features:` in the YAML. The `ColumnTransformer` will select them.
- **Categorical variables**: add another transformer in `src/pipelines.py` (e.g., `OneHotEncoder(handle_unknown="ignore")`) under a new `("cat", ...)` branch and list categorical columns in a separate config key.
- **Model alternatives**: swap `IsolationForest` for `LocalOutlierFactor` (batch only), `EllipticEnvelope`, or external libs (e.g., PyOD). Keep the preprocessor stage.
- **Logging**: replace `print` with Python `logging` or wire MLflow for experiment tracking.

---

## Troubleshooting
- **File not found**: run scripts from the **project root** or pass absolute paths to `--data`/`--config`.
- **Missing required features**: error like `Missing required features in DataFrame: [...]` → add columns to your CSV or update `features:` in the YAML.
- **Different sklearn versions**: models saved with a newer/older sklearn may not load. Check `iforest_meta.json` and align environments.
- **All predictions are 1 or -1**: revisit `contamination`; too small/large can mask structure. Also check scaling.

---

## Reproducibility
- We set `random_state` in config for deterministic training.
- Save the **trained pipeline** (not just params) to ensure identical preprocessing at inference time.

---


---

## Docker

Build the image from the repository root (where the `Dockerfile` lives):

```bash
# Build the image
docker build -t anomaly-iforest:latest .
```

Run **training** (mount local `data/` and `models/` into the container):

```bash
docker run --rm   -v "$PWD/data:/app/data"   -v "$PWD/models:/app/models"   anomaly-iforest:latest   scripts/train.py --data data/train.csv --config configs/config.yaml
```

Run **inference** (the scored CSV will be written back to your host):

```bash
docker run --rm   -v "$PWD/data:/app/data"   -v "$PWD/models:/app/models"   -v "$PWD:/app"   anomaly-iforest:latest   scripts/inference.py --data data/new_data.csv --config configs/config.yaml --out scored.csv
```


## References
- scikit‑learn IsolationForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- Outlier detection overview: https://scikit-learn.org/stable/modules/outlier_detection.html
- Plotly Express: https://plotly.com/python/plotly-express/


