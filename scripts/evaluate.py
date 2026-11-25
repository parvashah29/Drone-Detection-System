"""Evaluate weights on the validation split (mAP, precision, recall)."""
import argparse

import numpy as np

from common import ROOT, check_dataset, pick_device, resolve_weights, runtime_data_yaml


def main():
    ap = argparse.ArgumentParser(description="Evaluate a drone detector on the val split.")
    ap.add_argument("--weights", default=None, help="default: models/drone_best.pt")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--split", default="val")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of split to use")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    weights = resolve_weights(args.weights)
    check_dataset(splits=(args.split,))
    from ultralytics import YOLO
    model = YOLO(str(weights))
    m = model.val(data=str(runtime_data_yaml()), split=args.split, imgsz=args.imgsz,
                  batch=args.batch, fraction=args.fraction, device=pick_device(args.device),
                  project=str(ROOT / "runs"), name="evaluation", exist_ok=True, workers=2)
    print("\nModel Evaluation Metrics:")
    print(f"mAP@0.5:      {m.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {m.box.map:.3f}")
    print(f"Precision:    {float(np.mean(m.box.p)):.3f}")
    print(f"Recall:       {float(np.mean(m.box.r)):.3f}")


if __name__ == "__main__":
    main()
