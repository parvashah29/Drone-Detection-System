"""Fine-tune YOLOv8n on the drone dataset."""
import argparse

from common import ROOT, BASE_WEIGHTS, check_dataset, die, pick_device, runtime_data_yaml


def main():
    ap = argparse.ArgumentParser(description="Train a YOLOv8 drone detector.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--base", default=str(BASE_WEIGHTS),
                    help="starting weights (default: models/yolov8n.pt; auto-downloaded by "
                         "ultralytics as 'yolov8n.pt' if that file is missing)")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set to use")
    ap.add_argument("--project", default=str(ROOT / "training_output"))
    ap.add_argument("--name", default="drone_detector")
    ap.add_argument("--device", default=None, help="mps, cpu, 0 ... (default: auto)")
    args = ap.parse_args()

    check_dataset()
    from pathlib import Path
    from ultralytics import YOLO
    base = args.base if Path(args.base).is_file() else "yolov8n.pt"
    model = YOLO(base)
    model.train(data=str(runtime_data_yaml()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, patience=args.patience, fraction=args.fraction,
                project=args.project, name=args.name, exist_ok=True,
                device=pick_device(args.device), workers=2)
    print(f"Training done. Weights in {Path(args.project) / args.name / 'weights'}")


if __name__ == "__main__":
    main()
