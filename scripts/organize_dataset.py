"""Split a flat folder of image + YOLO label (.txt) pairs into train/val."""
import argparse
import random
import shutil
from pathlib import Path

from common import DATASET_DIR, IMAGE_EXTS, die


def find_pairs(source):
    """Return [(image, label)] for images in `source` (recursive) that have a same-named .txt
    next to them or in a sibling labels/ folder. Images without labels are skipped."""
    pairs = []
    for img in sorted(Path(source).rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        for lab in (img.with_suffix(".txt"), img.parent.parent / "labels" / (img.stem + ".txt")):
            if lab.is_file():
                pairs.append((img, lab))
                break
    return pairs


def split_pairs(pairs, val_fraction=0.2, seed=42):
    """Deterministic shuffle split into (train, val)."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n_val = int(round(len(pairs) * val_fraction))
    if len(pairs) > 1:
        n_val = min(max(n_val, 1), len(pairs) - 1)
    return pairs[n_val:], pairs[:n_val]


def main():
    ap = argparse.ArgumentParser(description="Create train/val folders from image+label pairs.")
    ap.add_argument("--source", default=str(DATASET_DIR / "dataset_txt"),
                    help="folder with images and same-named .txt labels")
    ap.add_argument("--dest", default=str(DATASET_DIR))
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if not Path(a.source).is_dir():
        die(f"source folder not found: {a.source}")
    pairs = find_pairs(a.source)
    if not pairs:
        die(f"no image/label pairs found under {a.source}")
    train, val = split_pairs(pairs, a.val_fraction, a.seed)
    for name, items in (("train", train), ("val", val)):
        for sub in ("images", "labels"):
            (Path(a.dest) / name / sub).mkdir(parents=True, exist_ok=True)
        for img, lab in items:
            shutil.copy2(img, Path(a.dest) / name / "images" / img.name)
            shutil.copy2(lab, Path(a.dest) / name / "labels" / lab.name)
    print(f"train: {len(train)} pairs, val: {len(val)} pairs -> {a.dest}")


if __name__ == "__main__":
    main()
