"""Resize every image in a folder (optional; YOLO also resizes on the fly)."""
import argparse
from pathlib import Path

import cv2

from common import die, list_images


def preprocess_images(input_dir, output_dir, size=(416, 416)):
    """Resize images from input_dir into output_dir. Returns the number written."""
    imgs = list_images(input_dir)
    if not imgs:
        die(f"no images found in {input_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n = 0
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            print(f"skipping unreadable image {p.name}")
            continue
        cv2.imwrite(str(Path(output_dir) / p.name), cv2.resize(img, size))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Resize a folder of images.")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--size", type=int, default=416)
    a = ap.parse_args()
    print(f"Wrote {preprocess_images(a.input_dir, a.output_dir, (a.size, a.size))} images")


if __name__ == "__main__":
    main()
