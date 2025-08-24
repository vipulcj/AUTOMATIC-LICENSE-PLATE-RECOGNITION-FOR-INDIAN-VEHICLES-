
import json
import pandas as pd
import numpy as np
from util import parse_bbox, _similar_plates


def interpolate_bboxes(df):
    """Interpolate missing bounding boxes and keep best license plate number by score."""
    new_rows = []
    grouped = df.groupby("car_id")

    for car_id, group in grouped:
        group = group.sort_values("frame_nmr")
        frames = group["frame_nmr"].astype(int).tolist()

        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i + 1]
            row1, row2 = group.iloc[i], group.iloc[i + 1]

            # Keep exact first row
            new_rows.append(row1.to_dict())

            if f2 == f1 + 1:
                continue

            # Parse bboxes
            car1 = parse_bbox(row1["car_bbox"])
            car2 = parse_bbox(row2["car_bbox"])
            plate1 = parse_bbox(row1["license_plate_bbox"])
            plate2 = parse_bbox(row2["license_plate_bbox"])

            # Pick best OCR text by score
            s1 = pd.to_numeric(pd.Series([row1["license_number_score"]]), errors="coerce").fillna(0).iloc[0]
            s2 = pd.to_numeric(pd.Series([row2["license_number_score"]]), errors="coerce").fillna(0).iloc[0]
            num1 = str(row1["license_number"]) if pd.notna(row1["license_number"]) else ""
            num2 = str(row2["license_number"]) if pd.notna(row2["license_number"]) else ""
            if s1 >= s2:
                best_num, best_score = num1, float(s1)
            else:
                best_num, best_score = num2, float(s2)

            # Interpolate gaps
            for f in range(f1 + 1, f2):
                ratio = (f - f1) / (f2 - f1)
                car_interp = np.array(car1) + ratio * (np.array(car2) - np.array(car1))
                plate_interp = np.array(plate1) + ratio * (np.array(plate2) - np.array(plate1))

                ps1 = pd.to_numeric(pd.Series([row1["license_plate_bbox_score"]]), errors="coerce").fillna(0).iloc[0]
                ps2 = pd.to_numeric(pd.Series([row2["license_plate_bbox_score"]]), errors="coerce").fillna(0).iloc[0]
                plate_score_interp = float(ps1 + ratio * (ps2 - ps1))

                new_rows.append({
                    "frame_nmr": int(f),
                    "car_id": int(car_id),
                    "car_bbox": json.dumps(list(map(float, car_interp))),
                    "license_plate_bbox": json.dumps(list(map(float, plate_interp))),
                    "license_plate_bbox_score": plate_score_interp,
                    "license_number": best_num,
                    "license_number_score": best_score,
                })

        # Add final row
        new_rows.append(group.iloc[-1].to_dict())

    return pd.DataFrame(new_rows)



def merge_car_ids(df, iou_threshold=0.85):
    """Merge car_ids that belong to the same physical car using plate similarity + IoU."""
    def iou(box1, box2):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        xi1, yi1 = max(x1, x1_), max(y1, y1_)
        xi2, yi2 = min(x2, x2_), min(y2, y2_)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        union = box1_area + box2_area - inter_area
        return inter_area / union if union > 0 else 0.0

    if df.empty:
        return df

    required_cols = {
        "frame_nmr", "car_id", "car_bbox",
        "license_plate_bbox", "license_plate_bbox_score",
        "license_number", "license_number_score"
    }
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return df

    df["license_number_score"] = pd.to_numeric(df["license_number_score"], errors="coerce").fillna(0)

    car_id_map = {}
    unique_ids = df["car_id"].unique()
    next_id = 0

    for cid in unique_ids:
        subdf = df[df["car_id"] == cid]
        if subdf.empty:
            continue

        idx = subdf["license_number_score"].idxmax()
        best_row = subdf.loc[idx]

        try:
            best_box = parse_bbox(best_row["car_bbox"])
        except Exception:
            df.loc[df["car_id"] == cid, "car_id"] = next_id
            car_id_map[next_id] = (str(best_row.get("license_number", "")), [0, 0, 0, 0])
            next_id += 1
            continue

        best_num = str(best_row.get("license_number", ""))
        best_score = float(best_row.get("license_number_score", 0))

        matched = False
        for existing_id, (ref_num, ref_box) in car_id_map.items():
            # dynamic similarity threshold based on OCR confidence
            if best_score >= 0.6:
                plate_threshold = 0.9  # stricter when OCR confident
            else:
                plate_threshold = 0.7  # looser when OCR uncertain

            if _similar_plates(best_num, ref_num, threshold=plate_threshold) and iou(best_box, ref_box) > iou_threshold:
                df.loc[df["car_id"] == cid, "car_id"] = existing_id
                matched = True
                break

        if not matched:
            car_id_map[next_id] = (best_num, best_box)
            df.loc[df["car_id"] == cid, "car_id"] = next_id
            next_id += 1

    return df




def main(input_csv: str = "test.csv", output_csv: str = "test_interpolated.csv"):
    print(f"[INFO] Loading {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except Exception:
        print("[INFO] No license plates detected. Empty CSV. Exiting add_missing_data.")
        # Still write an empty file with header if needed
        empty_cols = [
            'frame_nmr','car_id','car_bbox','license_plate_bbox',
            'license_plate_bbox_score','license_number','license_number_score'
        ]
        pd.DataFrame(columns=empty_cols).to_csv(output_csv, index=False)
        return

    if df.empty:
        print("[INFO] No license plates detected. Empty CSV. Exiting add_missing_data.")
        df.to_csv(output_csv, index=False)
        return

    print("[INFO] Interpolating missing data...")
    new_df = interpolate_bboxes(df)

    if new_df.empty:
        print("[INFO] Nothing to interpolate. Saving empty output.")
        new_df.to_csv(output_csv, index=False)
        return

    print("[INFO] Merging duplicate car IDs...")
    new_df = merge_car_ids(new_df)

    new_df = new_df.sort_values(["car_id", "frame_nmr"])
    new_df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved -> {output_csv}")


if __name__ == "__main__":
    main()
