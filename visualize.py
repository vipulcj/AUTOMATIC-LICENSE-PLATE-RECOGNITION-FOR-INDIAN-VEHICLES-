import cv2
import numpy as np
import pandas as pd
from util import parse_bbox


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=6, ratio=0.15):
    x1, y1 = top_left
    x2, y2 = bottom_right
    w, h = x2 - x1, y2 - y1
    lx = int(w * ratio)
    ly = int(h * ratio)

    # corners
    cv2.line(img, (x1, y1), (x1 + lx, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + ly), color, thickness)
    cv2.line(img, (x2, y1), (x2 - lx, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + ly), color, thickness)
    cv2.line(img, (x1, y2), (x1 + lx, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - ly), color, thickness)
    cv2.line(img, (x2, y2), (x2 - lx, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - ly), color, thickness)
    return img


def _focus_score(bgr_crop):
    if bgr_crop is None or bgr_crop.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = bgr_crop if len(bgr_crop.shape) == 2 else None
    if gray is None or gray.size == 0:
        return 0.0
    v = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(max(0.0, min(1.0, 1.0 - np.exp(-v / 300.0))))  # smooth 0..1


def main(video_path: str, input_csv: str = './test_interpolated.csv', output_path: str = './output.mp4'):
    results = pd.read_csv(input_csv)

    if results.empty:
        print("[INFO] Visualization skipped: no license plates detected.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Could not open video file.")
        raise SystemExit

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total video frames: {total_frames}")

    # ---- precompute best text & crop per car_id
    license_plate = {}
    unique_ids = np.unique(results['car_id'])

    # Clip helper
    clip01 = lambda x: float(max(0.0, min(1.0, x)))

    for car_id in unique_ids:
        car_rows = results[results['car_id'] == car_id]
        if car_rows.empty:
            continue

        # Select candidate rows by combined score (text + small weight * bbox)
        car_rows = car_rows.copy()
        car_rows["lns_norm"] = car_rows["license_number_score"].astype(float).clip(lower=0.0, upper=1.0)
        car_rows["lpb_norm"] = car_rows["license_plate_bbox_score"].astype(float).clip(lower=0.0, upper=1.0)
        car_rows["combo"] = car_rows["lns_norm"] + 0.15 * car_rows["lpb_norm"]
        car_rows = car_rows.sort_values("combo", ascending=False)

        # Try up to top 5 candidates; pick the sharpest among those with similar combo
        top = car_rows.head(5)

        best_crop = None
        best_text = ""
        best_text_score = 0.0
        best_focus = -1.0

        max_combo = top["combo"].iloc[0] if not top.empty else 0.0
        for _, row in top.iterrows():
            # Only consider rows reasonably close to the best combo
            if row["combo"] < max_combo * 0.85:
                continue

            frame_idx = int(row['frame_nmr'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            try:
                x1, y1, x2, y2 = parse_bbox(row['license_plate_bbox'])
                x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                x1i = max(0, min(x1i, frame.shape[1] - 1))
                x2i = max(0, min(x2i, frame.shape[1] - 1))
                y1i = max(0, min(y1i, frame.shape[0] - 1))
                y2i = max(0, min(y2i, frame.shape[0] - 1))
                if x2i <= x1i or y2i <= y1i:
                    continue
                crop = frame[y1i:y2i, x1i:x2i, :]
                if crop.size == 0:
                    continue
                fscore = _focus_score(crop)
                if fscore > best_focus:
                    best_focus = fscore
                    best_crop = crop
                    best_text = str(row['license_number'])
                    best_text_score = float(row['license_number_score'])
            except Exception as e:
                print(f"[ERROR] Failed processing car_id {car_id}: {e}")
                continue

        # Fallback: if all candidates failed, just pick the highest bbox score frame
        if best_crop is None:
            bbox_idx = car_rows['license_plate_bbox_score'].astype(float).idxmax()
            row = results.loc[bbox_idx]
            frame_idx = int(row['frame_nmr'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                try:
                    x1, y1, x2, y2 = parse_bbox(row['license_plate_bbox'])
                    x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                    x1i = max(0, min(x1i, frame.shape[1] - 1))
                    x2i = max(0, min(x2i, frame.shape[1] - 1))
                    y1i = max(0, min(y1i, frame.shape[0] - 1))
                    y2i = max(0, min(y2i, frame.shape[0] - 1))
                    if x2i > x1i and y2i > y1i:
                        crop = frame[y1i:y2i, x1i:x2i, :]
                        if crop.size > 0:
                            best_crop = crop
                            best_text = str(row['license_number'])
                            best_text_score = float(row['license_number_score'])
                except Exception as e:
                    print(f"[ERROR] Failed (fallback) processing car_id {car_id}: {e}")

        if best_crop is not None:
            license_plate[int(car_id)] = {
                'license_crop': best_crop,
                'license_plate_number': best_text,
                'license_plate_score': best_text_score
            }

    # ---- reset and render
    frame_nmr = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        frame_nmr += 1
        if not ret:
            break

        df_ = results[results['frame_nmr'] == frame_nmr]

        for _, row in df_.iterrows():
            try:
                car_id = int(row['car_id'])
                if car_id not in license_plate:
                    continue

                car_x1, car_y1, car_x2, car_y2 = map(int, parse_bbox(row['car_bbox']))
                draw_border(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 6)

                x1, y1, x2, y2 = map(int, parse_bbox(row['license_plate_bbox']))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                plate_num = license_plate[car_id]['license_plate_number']
                plate_score = license_plate[car_id]['license_plate_score']
                license_crop = license_plate[car_id]['license_crop']
                if license_crop is None:
                    continue

                crop_height = max(50, int((car_y2 - car_y1) * 0.2))
                aspect = license_crop.shape[1] / license_crop.shape[0]
                crop_resized = cv2.resize(license_crop, (int(crop_height * aspect), crop_height))
                H, W, _ = crop_resized.shape

                text_bg_height = max(28, int(H * 0.45))
                text_bg_y1 = max(0, int(car_y1) - H - text_bg_height - 12)
                text_bg_y2 = text_bg_y1 + text_bg_height
                text_bg_x1 = max(0, int((car_x2 + car_x1 - W) // 2))
                text_bg_x2 = min(width, text_bg_x1 + W)

                overlay = frame.copy()
                cv2.rectangle(overlay, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (255, 255, 255), -1)
                frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

                display_text = f"{plate_num} ({plate_score:.2f})"
                (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
                text_x = text_bg_x1 + max(0, (W - tw) // 2)
                text_y = text_bg_y1 + (text_bg_height + th) // 2 - 4
                cv2.putText(frame, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)

                crop_y1 = text_bg_y2 + 4
                crop_y2 = min(height, crop_y1 + H)
                crop_x1 = text_bg_x1
                crop_x2 = min(width, crop_x1 + W)
                if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                    frame[crop_y1:crop_y2, crop_x1:crop_x2] = crop_resized[:crop_y2-crop_y1, :crop_x2-crop_x1]

            except Exception as e:
                print(f"[ERROR] Failed drawing overlays on frame {frame_nmr}: {e}")
                continue

        out.write(frame)

    out.release()
    cap.release()
    print(f"[INFO] Video saved to {output_path}")


if __name__ == "__main__":
    # Optional manual run:
    # main(video_path='sample2.mp4')
    pass
