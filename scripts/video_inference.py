"""Run the drone detector on a video file or webcam, writing an annotated video."""
import argparse
import time
from pathlib import Path

import cv2

from common import ROOT, die, has_display, pick_device, resolve_weights


def process_video(source, weights=None, conf=0.45, iou=0.7, out_path=None, show=False,
                  max_frames=0, device=None, imgsz=416):
    """Annotate a video. Returns stats dict. source may be a path or int (webcam index)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        die(f"could not open video source: {source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    if w == 0 or h == 0:
        die(f"video has no readable frames: {source}")

    from ultralytics import YOLO
    model = YOLO(str(resolve_weights(weights)))
    device = pick_device(device)

    writer = None
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            die(f"could not open output file for writing: {out_path}")
    show = show and has_display()

    frames = dets = 0
    frames_with_det = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
        n = len(r.boxes)
        frames += 1
        dets += n
        frames_with_det += bool(n)
        annotated = r.plot()
        if writer:
            writer.write(annotated)
        if show:
            cv2.imshow("Drone Detection (q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if max_frames and frames >= max_frames:
            break
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    elapsed = max(time.time() - t0, 1e-9)
    stats = {"frames": frames, "detections": dets, "frames_with_detection": frames_with_det,
             "fps": frames / elapsed, "output": str(out_path) if out_path else None}
    if frames == 0:
        print("warning: no frames could be read from the source")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Detect drones in a video or webcam stream.")
    ap.add_argument("source", help="video file path, or 'webcam'")
    ap.add_argument("--weights", default=None, help="default: models/drone_best.pt")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--out", default=None,
                    help="output video path (default: output/videos/<name>_detected.mp4)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--show", action="store_true", help="show live window (needs a display)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    webcam = args.source == "webcam"
    if not webcam and not Path(args.source).is_file():
        die(f"video not found: {args.source}")
    src = 0 if webcam else args.source
    out = None
    if not args.no_save and not webcam:
        out = args.out or str(ROOT / "output" / "videos" / f"{Path(args.source).stem}_detected.mp4")
    s = process_video(src, args.weights, args.conf, args.iou, out, args.show or webcam,
                      args.max_frames, args.device)
    print(f"Frames: {s['frames']}  detections: {s['detections']}  "
          f"frames with a detection: {s['frames_with_detection']}  speed: {s['fps']:.1f} FPS")
    if s["output"]:
        print(f"Saved: {s['output']}")


if __name__ == "__main__":
    main()
