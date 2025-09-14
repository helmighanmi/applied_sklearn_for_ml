import sys, subprocess
from pathlib import Path

def test_train_script(tmp_path, tmp_config, tmp_data_files):
    train_py = Path("scripts/train.py")   # run from project root
    assert train_py.exists(), "scripts/train.py not found"

    train_csv, _ = tmp_data_files
    res = subprocess.run(
        [sys.executable, str(train_py), "--data", str(train_csv), "--config", str(tmp_config)],
        cwd=Path.cwd(),
        capture_output=True, text=True, check=True
    )
    # Check model artifacts exist
    from src.utils import load_config
    cfg = load_config(tmp_config)
    assert Path(cfg["model_path"]).exists()
    assert Path(cfg["meta_path"]).exists()

def test_inference_script(tmp_path, tmp_config, tmp_data_files):
    infer_py = Path("scripts/inference.py")
    assert infer_py.exists(), "scripts/inference.py not found"

    _, new_csv = tmp_data_files
    out_csv = tmp_path / "scored.csv"
    res = subprocess.run(
        [sys.executable, str(infer_py), "--data", str(new_csv), "--config", str(tmp_config), "--out", str(out_csv)],
        cwd=Path.cwd(),
        capture_output=True, text=True, check=True
    )
    assert out_csv.exists()
