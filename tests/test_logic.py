import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import common  # noqa: E402
import organize_dataset as od  # noqa: E402


def test_list_images_filters_and_sorts(tmp_path):
    for n in ("b.jpg", "a.PNG", "notes.txt"):
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in common.list_images(tmp_path)] == ["a.PNG", "b.jpg"]
    assert common.list_images(tmp_path / "missing") == []
    assert common.list_images(tmp_path / "notes.txt") == []


def test_resolve_weights_missing_explicit_exits(tmp_path):
    with pytest.raises(SystemExit):
        common.resolve_weights(tmp_path / "nope.pt")


def test_check_dataset_missing_exits(tmp_path):
    with pytest.raises(SystemExit):
        common.check_dataset(tmp_path)


def test_runtime_data_yaml_absolute_path(tmp_path):
    for s in ("train", "val"):
        (tmp_path / s / "images").mkdir(parents=True)
        (tmp_path / s / "labels").mkdir(parents=True)
    import yaml
    out = common.runtime_data_yaml(tmp_path)
    cfg = yaml.safe_load(out.read_text())
    assert Path(cfg["path"]) == tmp_path.resolve()
    assert cfg["names"] == {0: "drone"}


def test_split_pairs_deterministic_and_disjoint():
    pairs = [(f"{i}.jpg", f"{i}.txt") for i in range(50)]
    tr, va = od.split_pairs(pairs, 0.2, seed=1)
    assert len(va) == 10 and len(tr) == 40
    assert not set(tr) & set(va)
    assert od.split_pairs(pairs, 0.2, seed=1) == (tr, va)


def test_find_pairs_skips_unlabelled(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "a.txt").write_text("0 .5 .5 .1 .1")
    (tmp_path / "b.jpg").write_bytes(b"x")
    assert [i.name for i, _ in od.find_pairs(tmp_path)] == ["a.jpg"]


WEIGHTS = common.BUNDLED_WEIGHTS
VAL = common.DATASET_DIR / "val" / "images"


@pytest.mark.skipif(not WEIGHTS.is_file() or not VAL.is_dir(), reason="weights/dataset absent")
def test_smoke_detects_drone(tmp_path):
    from ultralytics import YOLO
    img = sorted(VAL.iterdir())[0]
    r = YOLO(str(WEIGHTS)).predict(str(img), conf=0.25, device="cpu", verbose=False)[0]
    assert len(r.boxes) >= 1


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="weights absent")
def test_bundled_asset_detection():
    import cv2
    from ultralytics import YOLO
    img = sorted((ROOT / "assets").glob("*.jpg"))[0]
    r = YOLO(str(WEIGHTS)).predict(cv2.imread(str(img)), conf=0.25, device="cpu", verbose=False)[0]
    assert r.boxes is not None
