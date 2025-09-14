import pandas as pd
from src.pipelines import build_pipeline
from src.utils import ensure_features, annotate_with_iforest, load_config

def test_ensure_features_passes(tiny_df):
    ensure_features(tiny_df, ["NPHI", "RHOB"])

def test_ensure_features_raises(tiny_df):
    try:
        ensure_features(tiny_df, ["MISSING"])
        assert False, "Expected ValueError for missing feature"
    except ValueError:
        pass

def test_build_fit_predict(tiny_df):
    pipe = build_pipeline(["NPHI","RHOB"], contamination=0.1, random_state=42)
    pipe.fit(tiny_df)
    scores = pipe.decision_function(tiny_df)
    preds = pipe.predict(tiny_df)
    assert len(scores) == len(tiny_df)
    assert set(pd.Series(preds).unique()).issubset({1, -1})

def test_annotate_with_iforest(tiny_df):
    pipe = build_pipeline(["NPHI","RHOB"])
    pipe.fit(tiny_df)
    out = annotate_with_iforest(tiny_df, pipe, ["NPHI","RHOB"])
    assert {"anomaly_score","anomaly_pred","anomaly_flag"}.issubset(out.columns)
    assert out["anomaly_flag"].dtype == bool

def test_load_config(tmp_config):
    cfg = load_config(tmp_config)
    assert cfg["features"] == ["NPHI","RHOB"]
