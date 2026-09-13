"""
Image Centerline Extraction -> Feed into Drawing Pipeline
=========================================================
Uses OpenCV skeletonization (thinning) to extract single-pass 
centerlines directly. Prevents double-drawing and back-and-forth retracing.

Dependencies:
  pip install opencv-contrib-python numpy
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np


# ----------------------------------------------------------------------
# Step 1: Extract 1-pixel wide centerline strokes (Single Pass)
# ----------------------------------------------------------------------
def extract_centerline_strokes(image_path: str, threshold: int = 128, 
                                invert: bool = False, min_points: int = 5):
    """
    Load image -> binarize -> skeletonize to 1-pixel line -> extract single-pass strokes.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    thresh_type = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
    _, binary = cv2.threshold(img, threshold, 255, thresh_type)

    # Skeletonize image down to 1-pixel wide line
    skeleton = cv2.ximgproc.thinning(binary)

    # Find external contours along the single-pixel skeleton
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    strokes = []
    for c in contours:
        pts = c.reshape(-1, 2).astype(float)
        if len(pts) < min_points:
            continue

        # Prevent loop retracing: if start and end are close, form a clean loop
        dist_start_end = np.linalg.norm(pts[0] - pts[-1])
        if dist_start_end < 10.0:
            pts = np.vstack((pts, pts[0]))

        strokes.append(pts)

    return strokes


# ----------------------------------------------------------------------
# Step 2: Resample strokes for uniform point density
# ----------------------------------------------------------------------
def resample_stroke(stroke: np.ndarray, points_per_unit_length: float = 0.3,
                    min_points: int = 3, max_points: int = 60):
    if len(stroke) < 2:
        return stroke

    dists = np.sqrt(np.sum(np.diff(stroke, axis=0)**2, axis=1))
    total_length = np.sum(dists)

    if total_length < 1e-6:
        return stroke[:1]

    n = int(np.clip(total_length * points_per_unit_length, min_points, max_points))

    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    target_dists = np.linspace(0, total_length, n)

    resampled_x = np.interp(target_dists, cum_dists, stroke[:, 0])
    resampled_y = np.interp(target_dists, cum_dists, stroke[:, 1])

    return np.column_stack((resampled_x, resampled_y))


# ----------------------------------------------------------------------
# Step 3: Normalize strokes to [0, 1] unit square
# ----------------------------------------------------------------------
def normalize_strokes_to_unit_square(strokes: list, margin: float = 0.1):
    if not strokes:
        return []

    all_pts = np.vstack(strokes)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    src_w, src_h = max(x_max - x_min, 1e-6), max(y_max - y_min, 1e-6)

    usable = 1.0 - 2 * margin
    scale = usable / max(src_w, src_h)

    normalized = []
    for stroke in strokes:
        x_norm = (stroke[:, 0] - x_min) * scale + margin
        y_norm = 1.0 - ((stroke[:, 1] - y_min) * scale + margin)  # Flip Y for whiteboard
        normalized.append([(round(float(x), 4), round(float(y), 4)) for x, y in zip(x_norm, y_norm)])

    return normalized


def export_strokes_for_teammate(normalized_strokes: list, out_json_path: str):
    with open(out_json_path, "w") as f:
        json.dump({"strokes": normalized_strokes}, f, indent=2)
    return out_json_path


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_to_svg_vectorize_en.py <path_to_image>")
        sys.exit(1)

    input_image = sys.argv[1]

    if not Path(input_image).exists():
        print(f"Error: file not found -> {input_image}")
        sys.exit(1)

    # 1. Extract 1-pixel centerline strokes
    raw_strokes = extract_centerline_strokes(input_image, threshold=128, invert=False)
    print(f"Extracted {len(raw_strokes)} continuous centerline stroke(s)")

    # 2. Resample points
    resampled_strokes = [resample_stroke(s) for s in raw_strokes]

    # 3. Normalize coordinates
    normalized = normalize_strokes_to_unit_square(resampled_strokes, margin=0.1)

    out_json = str(Path(input_image).with_suffix("")) + "_strokes.json"
    export_strokes_for_teammate(normalized, out_json)
    print(f"Exported normalized strokes: {out_json}")

    print("\nstrokes = [")
    for stroke in normalized:
        print(f"    {stroke},")
    print("]")