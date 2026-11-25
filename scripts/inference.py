"""Run the drone detector on an image or a folder of images."""
import argparse
from pathlib import Path

import cv2

from common import ROOT, DATASET_DIR, die, has_display, list_images, pick_device, resolve_weights


def annotate_image(model, img_path, conf, iou, device, out_dir):
    """Detect on one image, save annotated copy, return list of (conf, xyxy)."""
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"warning: cannot read image {img_path}, skipping")
        return None
    r = model.predict(frame, conf=conf, iou=iou, imgsz=416, device=device, verbose=False)[0]
    out = Path(out_dir) / f"pred_{Path(img_path).stem}.jpg"
    cv2.imwrite(str(out), r.plot())
    return [(float(c), [float(v) for v in b]) for c, b in zip(r.boxes.conf, r.boxes.xyxy)], out


def main():
    ap = argparse.ArgumentParser(description="Detect drones in images.")
    ap.add_argument("source", nargs="?", default=str(DATASET_DIR / "val" / "images"),
                    help="image file or folder (default: dataset val images)")
    ap.add_argument("--weights", default=None, help="default: models/drone_best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--out", default=str(ROOT / "detection_results"))
    ap.add_argument("--limit", type=int, default=0, help="max images to process (0 = all)")
    ap.add_argument("--show", action="store_true", help="show window (needs a display)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    images = list_images(args.source)
    if not images:
        die(f"no images found at {args.source}")
    if args.limit:
        images = images[:args.limit]
    weights = resolve_weights(args.weights)
    from ultralytics import YOLO
    model = YOLO(str(weights))
    device = pick_device(args.device)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    total = 0
    for p in images:
        res = annotate_image(model, p, args.conf, args.iou, device, args.out)
        if res is None:
            continue
        dets, out = res
        total += len(dets)
        print(f"{p.name}: {len(dets)} detection(s)" +
              "".join(f"  conf={c:.2f} box={[round(v) for v in b]}" for c, b in dets))
        if args.show and has_display():
            cv2.imshow("Drone Detection", cv2.imread(str(out)))
            cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Done: {len(images)} image(s), {total} detection(s). Results in {args.out}")


if __name__ == "__main__":
    main()
