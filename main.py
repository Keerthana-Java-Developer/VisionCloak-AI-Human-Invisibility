"""VisionCloak - real-time human invisibility effect using webcam segmentation."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from visioncloak.app import VisionCloakApp
from visioncloak.config import AppConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a real-time human invisibility effect with a webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Human segmentation confidence threshold, from 0.0 to 1.0"
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="Do not mirror the webcam preview"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Folder for snapshots and recordings"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not 0.0 <= args.threshold <= 1.0:
        print("Error: --threshold must be between 0.0 and 1.0.")
        return 2

    config = AppConfig(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        segmentation_threshold=args.threshold,
        mirror_preview=not args.no_mirror,
        output_dir=args.output_dir,
    )

    try:
        VisionCloakApp(config).run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logging.exception("VisionCloak stopped unexpectedly")
        print(f"\nVisionCloak could not start: {exc}")
        print("Check the Troubleshooting section in README.md.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
