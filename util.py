
import re
import csv
import json
import cv2
import easyocr
import numpy as np

# ---------------- OCR setup ----------------
# Keep GPU disabled by default to match most user environments.
# If you have CUDA and want to enable it, change gpu=False -> gpu=True
reader = easyocr.Reader(['en'], gpu=False)

# ---------------- Heuristics for plate cleanup ----------------
char_to_num = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'H': 'N'}
num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}


def strict_format(text: str) -> bool:
    """
    Indian plate format: 9–10 chars like MH12AB1234 or MH12A1234.
    """
    if text is None:
        return False
    if not (9 <= len(text) <= 10):
        return False
    if not (text[0].isalpha() and text[1].isalpha()):
        return False
    if not (text[2].isdigit() and text[3].isdigit()):
        return False
    if not text[4].isalpha():
        return False
    if len(text) == 10 and not text[5].isalpha():
        return False
    if len(text) == 9 and not text[5].isdigit():
        return False
    if not all(ch.isdigit() for ch in text[-4:]):
        return False
    return True


def clean_and_correct(text: str) -> str:
    """
    Keep only uppercase A-Z and digits; map common confusions
    depending on expected position (letters/digits).
    """
    if text is None:
        return ""
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(text) > 10:
        text = text[:10]

    text_list = list(text)
    for idx, ch in enumerate(text_list):
        # letters expected at indices 0,1,4,(5 if len==10); digits elsewhere
        if idx in [0, 1, 4, 5] and idx < len(text_list):
            if ch.isdigit():
                text_list[idx] = num_to_char.get(ch, ch)
        else:
            if ch.isalpha():
                text_list[idx] = char_to_num.get(ch, ch)
    return ''.join(text_list)


def read_license_plate(lp_img):
    """
    Strict-only OCR:
      - Accepts a color crop (lp_img).
      - Resizes to a reasonable OCR size.
    Returns: (best_text, best_score)
    """
    if lp_img is None:
        return None, None

    try:
        if len(lp_img.shape) == 2:
            color_raw = cv2.cvtColor(lp_img, cv2.COLOR_GRAY2BGR)
            gray_base = lp_img
        else:
            color_raw = lp_img.copy()
            gray_base = cv2.cvtColor(color_raw, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None, None

    h, w = gray_base.shape[:2]
    if w == 0 or h == 0:
        return None, None

    target_w = 240
    scale = target_w / float(w)
    target_h = max(24, int(h * scale))
    try:
        raw_resized = cv2.resize(color_raw, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        gray_resized = cv2.cvtColor(raw_resized, cv2.COLOR_BGR2GRAY)
    except Exception:
        raw_resized = color_raw
        gray_resized = gray_base

    best_text = None
    best_score = -1.0

    dets = reader.readtext(gray_resized)

    merged = "".join([d[1] for d in dets]).upper()
    scores = [d[2] for d in dets if len(d) >= 3]
    mean_score = float(sum(scores) / len(scores)) if scores else 0.0

    cleaned = clean_and_correct(merged)
    if strict_format(cleaned):
        if mean_score > best_score:
            best_score = mean_score
            best_text = cleaned

    if best_text is None:
        return None, None
    return best_text, float(best_score)


def get_car(lp, vehicle_track_ids):
    """
    Match a plate detection to a tracked vehicle.
    lp: (x1,y1,x2,y2,score,class_id) — as produced by YOLO boxes.data.tolist()
    vehicle_track_ids: iterable of (x1,y1,x2,y2,track_id) from SORT
    Returns matched car bbox + track_id or (-1,...)
    """
    try:
        x1, y1, x2, y2, score, class_id = lp
    except Exception:
        return -1, -1, -1, -1, -1

    # Use plate center containment to match to track boxes (more robust than strict inside)
    cx = (float(x1) + float(x2)) / 2.0
    cy = (float(y1) + float(y2)) / 2.0

    for item in vehicle_track_ids:
        try:
            xcar1, ycar1, xcar2, ycar2, car_id = item
        except Exception:
            continue
        if (cx >= xcar1) and (cx <= xcar2) and (cy >= ycar1) and (cy <= ycar2):
            return int(xcar1), int(ycar1), int(xcar2), int(ycar2), int(car_id)

    return -1, -1, -1, -1, -1


# Levenshtein & similarity
def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = prev[j] + 1
            dele = curr[j - 1] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def _similar_plates(p1: str, p2: str, threshold: float = 0.85) -> bool:
    if not p1 or not p2:
        return False
    if p1 == p2:
        return True
    if len(p1) == len(p2) and _levenshtein(p1, p2) <= 1:
        return True
    dist = _levenshtein(p1, p2)
    maxlen = max(len(p1), len(p2))
    ratio = 1.0 - (dist / maxlen)
    return ratio >= threshold


def write_csv(results, output_path, unify_by_plate: bool = True, similarity_threshold: float = 0.85):
    """
    CSV columns:
    frame_nmr, car_id, car_bbox, license_plate_bbox, license_plate_bbox_score, license_number, license_number_score

    Writes only rows whose license_number passes strict_format().
    Returns: number of data rows written (excluding header).
    """
    plate_to_gid = {}
    sort_to_gid = {}
    next_gid = 0
    row_count = 0

    frame_keys = sorted(results.keys(), key=lambda x: int(x) if str(x).isdigit() else x)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame_nmr',
            'car_id',
            'car_bbox',
            'license_plate_bbox',
            'license_plate_bbox_score',
            'license_number',
            'license_number_score'
        ])

        for frame_nmr in frame_keys:
            cars = results[frame_nmr]
            for sort_car_id, data in cars.items():
                if 'car' not in data or 'license_plates' not in data:
                    continue

                car_bbox = data['car'].get('bbox', [0, 0, 0, 0])
                lp = data['license_plates']
                plate_text = lp.get('text', None)
                text_score = float(lp.get('text_score', 0))
                lp_bbox = lp.get('bbox', [0, 0, 0, 0])
                lp_bbox_score = float(lp.get('bbox_score', 0))

                # Strict-only: skip row if text not valid
                if not plate_text or not strict_format(plate_text):
                    continue

                # Determine global ID (unify identical/near-identical plates)
                if unify_by_plate and plate_text:
                    assigned_gid = None
                    for known_plate, gid in plate_to_gid.items():
                        if _similar_plates(plate_text, known_plate, similarity_threshold):
                            assigned_gid = gid
                            break
                    if assigned_gid is None:
                        assigned_gid = next_gid
                        plate_to_gid[plate_text] = assigned_gid
                        next_gid += 1
                else:
                    if sort_car_id not in sort_to_gid:
                        sort_to_gid[sort_car_id] = next_gid
                        next_gid += 1
                    assigned_gid = sort_to_gid[sort_car_id]

                # JSON arrays to avoid CSV parsing issues
                car_bbox_json = json.dumps([float(x) for x in car_bbox])
                lp_bbox_json = json.dumps([float(x) for x in lp_bbox])

                writer.writerow([
                    int(frame_nmr),
                    int(assigned_gid),
                    car_bbox_json,
                    lp_bbox_json,
                    lp_bbox_score,
                    plate_text,
                    text_score
                ])
                row_count += 1

    return row_count


# ---- shared bbox parser (robust)
def parse_bbox(bbox_val):
    """
    Convert various bbox formats to [x1, y1, x2, y2] (floats).

    Accepts:
      - JSON string like "[x1, y1, x2, y2]"
      - String with spaces/commas and extra junk, e.g. "[x1 y1 x2 y2]"
      - Python list/tuple/np.ndarray of length 4
    """
    # Already a sequence?
    if isinstance(bbox_val, (list, tuple, np.ndarray)):
        if len(bbox_val) == 4:
            return [float(v) for v in bbox_val]
        raise ValueError(f"Invalid bbox sequence length: {bbox_val}")

    # NaN / empty
    if bbox_val is None:
        raise ValueError("Empty bbox value")

    # String?
    if isinstance(bbox_val, str):
        s = bbox_val.strip()
        if not s:
            raise ValueError("Empty bbox string")

        # Try JSON parse first
        try:
            obj = json.loads(s)
            if isinstance(obj, (list, tuple)) and len(obj) == 4:
                return [float(v) for v in obj]
        except Exception:
            pass

        # Fallback: pull first four numbers via regex (handles stray chars)
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if len(nums) >= 4:
            return [float(x) for x in nums[:4]]

        raise ValueError(f"Invalid bbox format: {bbox_val}")

    # If it’s a plain number, it's malformed data
    raise ValueError(f"Invalid bbox type/value: {type(bbox_val)} {bbox_val}")
