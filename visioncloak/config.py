"""Application configuration."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    segmentation_threshold: float = 0.65
    mirror_preview: bool = True
    output_dir: Path = Path("output")
    window_name: str = "VisionCloak | AI Human Invisibility"
