
"""Advanced Human and Shadow Removal."""
import cv2
import numpy as np


def apply_invisibility_effect(
    live_frame,
    background_frame,
    human_mask
):

    height, width = live_frame.shape[:2]

    if background_frame.shape[:2] != (height, width):
        background_frame = cv2.resize(
            background_frame, (width, height)
        )

    if human_mask.shape[:2] != (height, width):
        human_mask = cv2.resize(
            human_mask.astype(np.float32),
            (width, height)
        )

    # STEP 1: Detect the complete human region
    mask = np.clip(human_mask, 0, 1)

    binary_mask = (mask > 0.16).astype(np.uint8)

    # STEP 2: Fill small gaps
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (21, 21)
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    locations = cv2.findNonZero(binary_mask)

    if locations is None:
        return live_frame.copy()

    x, y, w, h = cv2.boundingRect(locations)

    # STEP 3: Expand human boundary
    expand = max(35, int(min(width, height) * 0.045))
    if expand % 2 == 0:
        expand += 1

    human_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (expand, expand)
    )

    expanded_mask = cv2.dilate(
        binary_mask,
        human_kernel,
        iterations=2
    )

    # STEP 4: Detect nearby dark shadows
    pad_x = int(w * 0.50)
    pad_top = int(h * 0.15)
    pad_bottom = int(h * 0.40)

    x1 = max(0, x - pad_x)
    x2 = min(width, x + w + pad_x)
    y1 = max(0, y - pad_top)
    y2 = min(height, y + h + pad_bottom)

    live_gray = cv2.cvtColor(
        live_frame, cv2.COLOR_BGR2GRAY
    ).astype(np.int16)

    bg_gray = cv2.cvtColor(
        background_frame, cv2.COLOR_BGR2GRAY
    ).astype(np.int16)

    difference = bg_gray - live_gray

    shadow_mask = np.zeros(
        (height, width), dtype=np.uint8
    )

    roi = difference[y1:y2, x1:x2]

    shadow_mask[y1:y2, x1:x2] = (
        roi > 10
    ).astype(np.uint8)

    shadow_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (15, 15)
    )

    shadow_mask = cv2.morphologyEx(
        shadow_mask,
        cv2.MORPH_CLOSE,
        shadow_kernel,
        iterations=2
    )

    shadow_mask = cv2.dilate(
        shadow_mask,
        shadow_kernel,
        iterations=1
    )

    # STEP 5: Combine human and shadow
    final_mask = cv2.bitwise_or(
        expanded_mask,
        shadow_mask
    )

    # STEP 6: Smooth the outer boundary
    alpha = cv2.GaussianBlur(
        final_mask.astype(np.float32),
        (0, 0),
        sigmaX=2.0
    )

    alpha = np.clip(alpha, 0, 1)
    alpha = alpha[:, :, np.newaxis]

    # STEP 7: Replace with original clean background
    result = (
        live_frame.astype(np.float32) * (1 - alpha)
        +
        background_frame.astype(np.float32) * alpha
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def add_status_panel(frame, lines, panel_height=112):

    output = frame.copy()
    overlay = output.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (output.shape[1], panel_height),
        (16, 22, 32),
        -1
    )

    cv2.addWeighted(
        overlay, 0.78, output, 0.22, 0, output
    )

    y = 30

    for index, line in enumerate(lines):

        scale = 0.68 if index == 0 else 0.52
        thickness = 2 if index == 0 else 1

        color = (
            (120, 230, 170)
            if index == 0
            else (235, 240, 245)
        )

        cv2.putText(
            output,
            line,
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )

        y += 27

    return output
