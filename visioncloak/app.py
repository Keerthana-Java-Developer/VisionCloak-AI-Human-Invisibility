"""Webcam application and keyboard interaction."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig
from .effects import add_status_panel, apply_invisibility_effect
from .segmentation import HumanSegmenter

LOGGER = logging.getLogger(__name__)


class VisionCloakApp:
    """Runs the webcam preview, segmentation, and invisibility compositing."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.background: np.ndarray | None = None
        self.invisibility_enabled = False
        self.recording = False
        self.video_writer: cv2.VideoWriter | None = None
        self.segmenter = HumanSegmenter(model_selection=1)
        self.camera: cv2.VideoCapture | None = None
        self.last_frame: np.ndarray | None = None
        self.fps = 0.0

    def _open_camera(self) -> None:
        self.camera = cv2.VideoCapture(self.config.camera_index)
        if not self.camera.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.config.camera_index}. "
                "Check webcam permission or try --camera 1."
            )
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        LOGGER.info("Webcam opened")

    def _capture_background(self, frame: np.ndarray) -> None:
        self.background = frame.copy()
        path = self.output_dir / "captured_background.jpg"
        if not cv2.imwrite(str(path), self.background):
            LOGGER.warning("Could not save background image")
        else:
            LOGGER.info("Clean background captured: %s", path.resolve())
        self.invisibility_enabled = False

    def _save_snapshot(self, frame: np.ndarray) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"visioncloak_snapshot_{stamp}.jpg"
        cv2.imwrite(str(path), frame)
        LOGGER.info("Snapshot saved: %s", path.resolve())

    def _toggle_recording(self, frame: np.ndarray) -> None:
        if not self.recording:
            height, width = frame.shape[:2]
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.output_dir / f"visioncloak_recording_{stamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(path), fourcc, 20.0, (width, height))
            if not writer.isOpened():
                writer.release()
                LOGGER.error("Could not start video recording")
                return
            self.video_writer = writer
            self.recording = True
            LOGGER.info("Recording started: %s", path.resolve())
        else:
            self._stop_recording()

    def _stop_recording(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        LOGGER.info("Recording stopped")

    def _help_lines(self) -> list[str]:
        background_state = "READY" if self.background is not None else "NOT CAPTURED"
        effect_state = "ON" if self.invisibility_enabled else "OFF"
        record_state = "REC" if self.recording else "IDLE"
        return [
            "VISIONCLOAK  |  AI HUMAN INVISIBILITY",
            f"Background: {background_state}   Effect: {effect_state}   Recording: {record_state}",
            "B: Capture empty background   I: Toggle effect   R: Reset   S: Snapshot   V: Record   Q: Quit",
        ]

    def run(self) -> None:
        self._open_camera()
        LOGGER.info("Press B with an EMPTY scene to capture the background.")
        try:
            while True:
                ok, frame = self.camera.read() if self.camera is not None else (False, None)
                if not ok or frame is None:
                    LOGGER.warning("Camera frame unavailable; retrying")
                    if cv2.waitKey(100) & 0xFF == ord("q"):
                        break
                    continue

                if self.config.mirror_preview:
                    frame = cv2.flip(frame, 1)
                self.last_frame = frame.copy()

                display = frame
                if self.invisibility_enabled and self.background is not None:
                    mask = self.segmenter.get_mask(
                        frame, self.config.segmentation_threshold
                    )
                    display = apply_invisibility_effect(frame, self.background, mask)

                display = add_status_panel(display, self._help_lines())
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(display)

                cv2.imshow(self.config.window_name, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("b") and self.last_frame is not None:
                    self._capture_background(self.last_frame)
                elif key == ord("i"):
                    if self.background is None:
                        LOGGER.warning("Capture a clean background first (press B).")
                    else:
                        self.invisibility_enabled = not self.invisibility_enabled
                        LOGGER.info("Invisibility effect %s",
                                    "enabled" if self.invisibility_enabled else "disabled")
                elif key == ord("r"):
                    self.background = None
                    self.invisibility_enabled = False
                    LOGGER.info("Background reset")
                elif key == ord("s") and self.last_frame is not None:
                    self._save_snapshot(display)
                elif key == ord("v"):
                    self._toggle_recording(display)
        finally:
            self._stop_recording()
            if self.camera is not None:
                self.camera.release()
            self.segmenter.close()
            cv2.destroyAllWindows()
            LOGGER.info("VisionCloak closed")
