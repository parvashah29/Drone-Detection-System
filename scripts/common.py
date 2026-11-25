"""Shared helpers: project paths, device selection, weights/dataset checks."""
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "drone_dataset_yolo"
DATA_YAML = ROOT / "data.yaml"
BUNDLED_WEIGHTS = ROOT / "models" / "drone_best.pt"
TRAINED_WEIGHTS = ROOT / "training_output" / "drone_detector" / "weights" / "best.pt"
BASE_WEIGHTS = ROOT / "models" / "yolov8n.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def die(msg):
    """Exit with a clear one-line error."""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def pick_device(requested=None):
    """Return 'mps' on Apple silicon, 'cuda' if available, else 'cpu'."""
    if requested:
        return requested
    import torch
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_weights(path=None):
    """Find weights: explicit path, else bundled model, else local training output."""
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            die(f"weights not found: {p}")
        return p
    for cand in (BUNDLED_WEIGHTS, TRAINED_WEIGHTS):
        if cand.is_file():
            return cand
    die("no weights found. Expected models/drone_best.pt (shipped in the repo) "
        "or training_output/drone_detector/weights/best.pt (after running scripts/train.py). "
        "Pass --weights PATH to use another file.")


def check_dataset(dataset_dir=DATASET_DIR, splits=("train", "val")):
    """Exit with a helpful message unless images/ and labels/ exist for each split."""
    dataset_dir = Path(dataset_dir)
    for s in splits:
        if not (dataset_dir / s / "images").is_dir() or not (dataset_dir / s / "labels").is_dir():
            die(f"dataset not found at {dataset_dir} (missing {s}/images or {s}/labels). "
                "See the 'Dataset' section of the README for the expected layout, "
                "or run scripts/organize_dataset.py.")
    return dataset_dir


def runtime_data_yaml(dataset_dir=DATASET_DIR, data_yaml=DATA_YAML):
    """Write a temp copy of data.yaml with an absolute dataset path and return its path."""
    check_dataset(dataset_dir)
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(Path(dataset_dir).resolve())
    out = Path(tempfile.mkdtemp(prefix="drone_data_")) / "data.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f)
    return out


def list_images(path):
    """Return a sorted list of image files for a file or directory."""
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix.lower() in IMAGE_EXTS else []
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    return []


def has_display():
    """True if a GUI window can plausibly be opened."""
    import os
    if sys.platform == "darwin" or sys.platform.startswith("win"):
        return not os.environ.get("SSH_CONNECTION")
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
