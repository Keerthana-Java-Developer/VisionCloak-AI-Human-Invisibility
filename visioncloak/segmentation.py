
"""Human segmentation using MediaPipe Selfie Segmentation."""

import cv2
import mediapipe as mp
import numpy as np


class HumanSegmenter:
    """Detects the human region in a webcam frame."""

    def __init__(self, model_selection=1):
        self.segmenter = (
            mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=model_selection
            )
        )

    def get_mask(self, frame, threshold=0.35):
        """
        Return a human segmentation mask.

        Args:
            frame: OpenCV BGR webcam frame.
            threshold: Confidence threshold for human detection.

        Returns:
            Float32 mask containing values between 0 and 1.
        """
        if frame is None:
            raise ValueError("Input frame cannot be None")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.segmenter.process(rgb_frame)

        if results.segmentation_mask is None:
            height, width = frame.shape[:2]
            return np.zeros((height, width), dtype=np.float32)

        mask = results.segmentation_mask.astype(np.float32)

        return (mask > threshold).astype(np.float32)

    def close(self):
        """Release MediaPipe resources."""
        if self.segmenter is not None:
            self.segmenter.close()
            self.segmenter = None
