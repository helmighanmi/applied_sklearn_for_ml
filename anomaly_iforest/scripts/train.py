# scripts/train.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from joblib import dump

from src.pipelines import build_pipeline
from src.utils import load_config, save_meta, ensure_features, default_meta


def main():
    ap = argparse.ArgumentParser(description="Train and save anomaly detection pipeline")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--data", required=True, help="Path to CSV for training")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Core config
    features = cfg["features"]
    random_state = int(cfg.get("random_state", 42))
    imputer_strategy = cfg.get("imputer_strategy", "median")
    scaler = cfg.get("scaler", "standard")

    # Model/artifacts paths
    model_path = Path(cfg.get("model_path", "models/iforest_pipeline.joblib"))
    meta_path = Path(cfg.get("meta_path", "models/iforest_meta.json"))

    # Algorithm selection (default: Isolation Forest)
    algo = cfg.get("algo", "iforest")
    algo_params = dict(cfg.get("algo_params", {}) or {})

    # Backward-compat: if 'contamination' is in cfg and not in algo_params, pass it for algos that use it
    contamination = cfg.get("contamination", None)
    if contamination is not None and "contamination" not in algo_params and algo.lower() in {"iforest", "elliptic", "pca"}:
        algo_params["contamination"] = float(contamination)

    # Load data
    df = pd.read_csv(args.data)
    ensure_features(df, features)

    # Build + fit pipeline
    pipe = build_pipeline(
        features=features,
        algo=algo,
        algo_params=algo_params,
        random_state=random_state,
        imputer_strategy=imputer_strategy,
        scaler=scaler,
    )
    pipe.fit(df)

    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, model_path)

    # Save metadata (augment default_meta with algo info)
    meta = default_meta(cfg, model_path)
    meta.update({
        "algo": algo,
        "algo_params": algo_params,
        "n_samples": int(len(df)),
    })
    save_meta(meta, meta_path)

    print(f"[OK] Trained '{algo}' on {len(df)} rows, features={features}")
    print(f"[OK] Model saved to {model_path}")
    print(f"[OK] Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
