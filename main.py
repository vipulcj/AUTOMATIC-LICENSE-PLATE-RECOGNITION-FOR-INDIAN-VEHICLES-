import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import *
from util import write_csv, get_car, read_license_plate
import add_missing_data
import visualize


def _focus_score(bgr_crop: np.ndarray) -> float:
    """
    Blur/Sharpness proxy: variance of Laplacian.
    Returns a normalized score ~[0,1].
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = bgr_crop if len(bgr_crop.shape) == 2 else None
    if gray is None or gray.size == 0:
        return 0.0
    v = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Normalize: saturate around ~600–800 variance
    return float(max(0.0, min(1.0, 1.0 - np.exp(-v / 300.0))))  # smooth 0..1


def run_pipeline(video_path: str, det_conf: float = 0.88, det_imgsz: int = 736):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # setup filenames
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    raw_csv = f"{video_name}_test.csv"
    interpolated_csv = f"{video_name}_interpolated.csv"
    output_video = f"{video_name}_output.mp4"

    results = {}
    mot_tracker = Sort()

    # load models
    coco_model = YOLO('yolov8n.pt')               # vehicle detection
    lpd_model  = YOLO('INPD_more_accuracy_n.pt')  # license plate detection

    # open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # COCO class ids for vehicles: car(2), motorcycle(3), bus(5), truck(7)
    vehicles = [2, 3, 5, 7]

    frame_no = -1

    # Track per-car best choice with combined score
    # {car_id: {"text": str, "text_score": float, "bbox": [..], "bbox_score": float, "combined": float}}
    best_for_car = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        results[frame_no] = {}

        H, W = frame.shape[:2]

        # vehicle detection
        det = coco_model(frame, conf=det_conf, imgsz=det_imgsz)[0]
        det_list = [
            [x1, y1, x2, y2, score]
            for x1, y1, x2, y2, score, class_id in det.boxes.data.tolist()
            if int(class_id) in vehicles
        ]
        det_arr = np.asarray(det_list)
        if det_arr.size == 0:
            det_arr = np.empty((0, 5))

        # tracking (SORT)
        track_ids = mot_tracker.update(det_arr)  # rows: x1,y1,x2,y2,track_id

        # license plate detection (global) and match to car
        lp_res = lpd_model(frame)[0]
        for lp in lp_res.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = lp

            # match plate to a tracked vehicle
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)
            if car_id == -1:
                continue

            # crop (with padding)
            pad_y = int((y2 - y1) * 0.15)
            pad_x = int((x2 - x1) * 0.10)
            x1p = max(0, int(x1 - pad_x))
            y1p = max(0, int(y1 - pad_y))
            x2p = min(frame.shape[1], int(x2 + pad_x))
            y2p = min(frame.shape[0], int(y2 + pad_y))
            lp_cropped = frame[y1p:y2p, x1p:x2p]
            if lp_cropped.size == 0:
                continue

            # OCR (strict-only inside util.read_license_plate)
            lp_text, lp_text_score = read_license_plate(lp_cropped)
            if lp_text is None:
                # strictly ignore invalid formats at OCR stage
                continue

            # Combined score = OCR + small weight * bbox confidence + small weight * sharpness
            sharp = _focus_score(lp_cropped)
            text_comp = max(0.0, min(1.0, float(lp_text_score)))
            det_comp  = max(0.0, min(1.0, float(score)))
            combined  = text_comp + 0.15 * det_comp + 0.10 * sharp

            prev = best_for_car.get(int(car_id))
            if prev is None or combined > prev["combined"]:
                best_for_car[int(car_id)] = {
                    "text": lp_text,
                    "text_score": float(lp_text_score),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "bbox_score": float(score),
                    "combined": float(combined)
                }

            # Store the (current) best-known info for this car at this frame
            chosen = best_for_car[int(car_id)]
            results[frame_no][int(car_id)] = {
                'car': {'bbox': [int(xcar1), int(ycar1), int(xcar2), int(ycar2)]},
                'license_plates': {
                    'bbox': chosen["bbox"],
                    'text': chosen["text"],
                    'bbox_score': chosen["bbox_score"],
                    'text_score': chosen["text_score"]
                }
            }

    cap.release()

    # write base CSV (strict-only rows)
    rows_written = write_csv(results, raw_csv)

    if rows_written == 0:
        # exit + keep the empty CSV with proper header
        print("[INFO] No license plates detected. Empty CSV written. Exiting.")
        return

    # interpolate + merge
    add_missing_data.main(input_csv=raw_csv, output_csv=interpolated_csv)

    # visualize
    visualize.main(video_path=video_path, input_csv=interpolated_csv, output_path=output_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Path to input video")
    parser.add_argument("--det-conf", type=float, default=0.88,
                        help="YOLO confidence threshold for VEHICLE detection (default: 0.88)")
    parser.add_argument("--det-imgsz", type=int, default=736,
                        help="YOLO input image size for VEHICLE detection (default: 736)")
    args = parser.parse_args()
    run_pipeline(args.video_path, det_conf=args.det_conf, det_imgsz=args.det_imgsz)
