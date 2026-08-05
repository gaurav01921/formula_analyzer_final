import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def segment_formula_lines(image_input, padding=12, min_line_height=18):
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input format.")

    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    img_h, img_w = gray.shape

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )

    row_sums = np.sum(thresh, axis=1)
    max_row = np.max(row_sums)

    if max_row == 0:
        return [pil_img]

    norm_profile = row_sums / max_row
    is_text = norm_profile > 0.03

    line_ranges = []
    in_line = False
    start_y = 0

    for y, active in enumerate(is_text):
        if active and not in_line:
            in_line = True
            start_y = y
        elif not active and in_line:
            in_line = False
            if (y - start_y) >= min_line_height:
                line_ranges.append((start_y, y))

    if in_line and (img_h - start_y) >= min_line_height:
        line_ranges.append((start_y, img_h))

    merged_ranges = []
    min_gap = int(img_h * 0.05)
    for r in line_ranges:
        if not merged_ranges:
            merged_ranges.append(r)
        else:
            prev_start, prev_end = merged_ranges[-1]
            curr_start, curr_end = r
            if (curr_start - prev_end) < min_gap:
                merged_ranges[-1] = (prev_start, curr_end)
            else:
                merged_ranges.append(r)

    if len(merged_ranges) <= 1:
        return [pil_img]

    crops = []
    for y1, y2 in merged_ranges:
        crop_y1 = max(0, y1 - padding)
        crop_y2 = min(img_h, y2 + padding)
        crop = pil_img.crop((0, crop_y1, img_w, crop_y2))
        crops.append(crop)

    return crops
