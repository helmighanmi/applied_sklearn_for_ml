# scripts/inference.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import plotly.express as px
from joblib import load

from src.utils import load_config, ensure_features, annotate_with_iforest


def main():
    ap = argparse.ArgumentParser(
        description="Run inference with trained IsolationForest pipeline"
    )
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--data", required=True, help="Path to CSV for scoring")
    ap.add_argument("--out", default="scored.csv")
    ap.add_argument("--plot", action="store_true", help="Show Plotly scatter after scoring")
    args = ap.parse_args()

    cfg = load_config(args.config)
    features = cfg["features"]
    model_path = Path(cfg.get("model_path", "models/iforest_pipeline.joblib"))

    df = pd.read_csv(args.data)
    ensure_features(df, features)

    pipe = load(model_path)

    df_annot = annotate_with_iforest(df, pipe, features)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_annot.to_csv(args.out, index=False)
    print(f"[OK] Wrote annotated data -> {args.out}")

    if args.plot:
        # Melt to tidy for a combined scatter, colored by variable
        long = (
            df_annot
            .reset_index(drop=False)
            .rename(columns={"index": "idx"})
            .melt(
                id_vars=["idx", "anomaly_flag", "anomaly_score"],
                value_vars=features,
                var_name="Variable",
                value_name="Value",
            )
            .dropna(subset=["Value"])
        )
        fig = px.scatter(
            long,
            x="idx",
            y="Value",
            color="Variable",
            symbol="anomaly_flag",
            hover_data=["anomaly_score"],
            title="Features (melted) with IsolationForest anomalies",
        )
        fig.show()


if __name__ == "__main__":
    main()
