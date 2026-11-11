
"""Human and cast-shadow removal using OpenCV."""

import cv2
import numpy as np


def apply_invisibility_effect(live_frame, background_frame, human_mask):
    if live_frame is None or background_frame is None:
        raise ValueError("Live frame and background frame are required")

    height, width = live_frame.shape[:2]

    # Match the captured background to the webcam dimensions
    background = cv2.resize(
        background_frame, (width, height)
    )

    # --------------------------------------------------
    # STEP 1: Prepare the MediaPipe human mask
    # --------------------------------------------------
    if human_mask is None:
        person_mask = np.zeros((height, width), dtype=np.uint8)
    else:
        mask = np.asarray(human_mask, dtype=np.float32)

        if mask.ndim == 3:
            mask = cv2.cvtColor(
                mask, cv2.COLOR_BGR2GRAY
            )

        mask = cv2.resize(mask, (width, height))

        if mask.max() > 1.0:
            mask = mask / 255.0

        person_mask = (mask > 0.35).astype(np.uint8)

    # Fill holes in the person mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (15, 15)
    )

    person_mask = cv2.morphologyEx(
        person_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Expand mask to cover the person's outline
    person_mask = cv2.dilate(
        person_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (31, 31)
        ),
        iterations=1
    )

    # --------------------------------------------------
    # STEP 2: Detect shadows independently of MediaPipe
    # --------------------------------------------------
    live_lab = cv2.cvtColor(
        live_frame, cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    background_lab = cv2.cvtColor(
        background, cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    # LAB channel 0 represents brightness
    live_brightness = live_lab[:, :, 0]
    background_brightness = background_lab[:, :, 0]

    # Shadows make the current image darker
    brightness_drop = (
        background_brightness - live_brightness
    )

    # Require a meaningful brightness reduction
    dark_threshold = np.maximum(
        10.0,
        background_brightness * 0.06
    )

    darker_pixels = brightness_drop > dark_threshold

    # --------------------------------------------------
    # STEP 3: Check colour similarity
    # Shadows normally change brightness more than colour.
    # LAB channels 1 and 2 represent colour information.
    # --------------------------------------------------
    colour_difference = np.sqrt(
        (background_lab[:, :, 1] - live_lab[:, :, 1]) ** 2
        +
        (background_lab[:, :, 2] - live_lab[:, :, 2]) ** 2
    )

    similar_colour = colour_difference < 18.0

    # Limit shadow detection to the area around the detected person.
    # This helps avoid replacing unrelated dark objects elsewhere in the scene.
    nearby_person_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (181, 181)
    )
    nearby_person_area = cv2.dilate(
        person_mask, nearby_person_kernel, iterations=1
    )

    shadow_mask = (
        darker_pixels
        & similar_colour
        & (nearby_person_area > 0)
    ).astype(np.uint8)

    # --------------------------------------------------
    # STEP 4: Clean the shadow mask
    # --------------------------------------------------
    shadow_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )

    shadow_mask = cv2.morphologyEx(
        shadow_mask,
        cv2.MORPH_CLOSE,
        shadow_kernel,
        iterations=2
    )

    shadow_mask = cv2.morphologyEx(
        shadow_mask,
        cv2.MORPH_OPEN,
        shadow_kernel,
        iterations=1
    )

    # --------------------------------------------------
    # STEP 5: Combine person and shadow masks
    # --------------------------------------------------
    final_mask = cv2.bitwise_or(
        person_mask, shadow_mask
    )

    # Remove tiny gaps in the combined mask
    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # --------------------------------------------------
    # STEP 6: Smooth replacement boundaries
    # --------------------------------------------------
    alpha = cv2.GaussianBlur(
        final_mask.astype(np.float32),
        (0, 0),
        sigmaX=1.5
    )

    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = alpha[:, :, np.newaxis]

    # --------------------------------------------------
    # STEP 7: Replace detected regions with clean background
    # --------------------------------------------------
    result = (
        live_frame.astype(np.float32) * (1.0 - alpha)
        +
        background.astype(np.float32) * alpha
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
