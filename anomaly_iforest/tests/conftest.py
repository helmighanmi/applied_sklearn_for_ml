import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import yaml

@pytest.fixture
def tiny_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "NPHI": rng.uniform(0.05, 0.35, 120),
        "RHOB": rng.normal(2.5, 0.1, 120)
    })

@pytest.fixture
def tmp_config(tmp_path: Path):
    cfg = {
        "features": ["NPHI", "RHOB"],
        "contamination": 0.1,
        "random_state": 42,
        "imputer_strategy": "median",
        "scaler": "standard",
        "model_path": str(tmp_path / "models" / "iforest_pipeline.joblib"),
        "meta_path": str(tmp_path / "models" / "iforest_meta.json"),
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path

@pytest.fixture
def tmp_data_files(tmp_path: Path, tiny_df: pd.DataFrame):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_csv = data_dir / "train.csv"
    new_csv = data_dir / "new_data.csv"
    tiny_df.to_csv(train_csv, index=False)
    tiny_df.sample(frac=1.0, random_state=0).to_csv(new_csv, index=False)
    return train_csv, new_csv
