from pathlib import Path
import pandas as pd
from joblib import dump, load
from src.pipelines import build_pipeline
from src.utils import load_config, ensure_features, annotate_with_iforest, default_meta, save_meta
from src.visualize import plot_scatter_melted, plot_score_hist, plot_pair_scatter, plot_feature_box

def test_end_to_end(tmp_path, tmp_config, tmp_data_files):
    cfg = load_config(tmp_config)
    features = cfg["features"]
    train_csv, new_csv = tmp_data_files

    df_train = pd.read_csv(train_csv)
    ensure_features(df_train, features)

    pipe = build_pipeline(
        features=features,
        contamination=float(cfg["contamination"]),
        random_state=int(cfg["random_state"]),
        imputer_strategy=cfg["imputer_strategy"],
        scaler=cfg["scaler"],
    )
    pipe.fit(df_train)

    model_path = Path(cfg["model_path"])
    meta_path = Path(cfg["meta_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, model_path)
    save_meta(default_meta(cfg, model_path), meta_path)

    assert model_path.exists() and meta_path.exists()

    # Inference
    pipe_loaded = load(model_path)
    df_new = pd.read_csv(new_csv)
    ensure_features(df_new, features)
    df_annot = annotate_with_iforest(df_new, pipe_loaded, features)
    assert {"anomaly_score","anomaly_pred","anomaly_flag"}.issubset(df_annot.columns)

    # Smoke-test visualizations (no UI)
    plot_scatter_melted(df_annot, features, show=False)
    plot_score_hist(df_annot, show=False)
    if len(features) >= 2:
        plot_pair_scatter(df_annot, x=features[1], y=features[0], show=False)
    plot_feature_box(df_annot, features, show=False)
