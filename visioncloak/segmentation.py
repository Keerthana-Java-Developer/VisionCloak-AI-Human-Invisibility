
"""AI Human Segmentation using MediaPipe."""
import cv2
import mediapipe as mp
import numpy as np


class HumanSegmenter:

    def __init__(self, model_selection=1):
        self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )

    def get_mask(self, frame_bgr, threshold=0.40):

        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty camera frame")

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.segmenter.process(rgb)

        if result.segmentation_mask is None:
            return np.zeros(frame_bgr.shape[:2], dtype=np.float32)

        mask = result.segmentation_mask.astype(np.float32)

        # Remove low-confidence noise
        mask[mask < threshold * 0.20] = 0

        return np.clip(mask, 0, 1)

    def close(self):
        self.segmenter.close()
