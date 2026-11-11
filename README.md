# VisionCloak — AI-Powered Human Invisibility

A real-time computer-vision demonstration that detects a person in a webcam stream and replaces the segmented person region with a clean background frame captured earlier.

**Important:** This is a software visual effect, not physical invisibility. It works best with a fixed camera, stable lighting, and a background captured while nobody is in the scene.

## Project highlights

- Real-time webcam video processing with OpenCV
- Human segmentation using MediaPipe Selfie Segmentation
- Soft mask refinement to reduce noisy edges
- Background compositing for the invisibility effect
- Background capture and reset controls
- Snapshot and MP4 recording support
- Configurable camera index, resolution, and segmentation threshold
- Modular code structure and a small unit test

## Demo workflow

1. Place the webcam or laptop on a stable surface.
2. Keep the camera view empty and press **B** to capture the clean background.
3. Enter the camera view.
4. Press **I** to enable the invisibility effect.
5. Press **I** again to disable it.

The captured background is saved to `output/captured_background.jpg`.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10 or 3.11, 64-bit recommended
- A working webcam
- Internet connection for the first dependency installation

Python 3.12+ may require different package versions. Use Python 3.11 for the most predictable setup with the pinned dependencies.

## Installation (Windows)

Open Command Prompt in the extracted `VisionCloak_AI_Human_Invisibility` folder.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

You can also double-click `run_windows.bat`. It installs the listed dependencies and starts the app using the `python` command available on your PATH.

## Installation (macOS / Linux)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## Controls

| Key | Action |
|---|---|
| `B` | Capture the current frame as the clean background. The scene must be empty. |
| `I` | Turn the invisibility effect on/off. Capture a background first. |
| `R` | Reset the captured background and turn the effect off. |
| `S` | Save a snapshot in the `output/` folder. |
| `V` | Start/stop MP4 recording in the `output/` folder. |
| `Q` | Quit safely. |

## Run options

```bash
python main.py --camera 0 --width 1280 --height 720 --threshold 0.65
```

- `--camera`: webcam index; try `1` if the default camera does not open.
- `--width`, `--height`: requested camera resolution.
- `--threshold`: segmentation confidence threshold from `0.0` to `1.0`. Increase it to reduce false detections; decrease it if parts of the person are missed.
- `--no-mirror`: show the camera without horizontal mirroring.
- `--output-dir`: choose another folder for snapshots and recordings.
- `--debug`: print detailed logs.

## Project structure

```text
VisionCloak_AI_Human_Invisibility/
├── main.py
├── requirements.txt
├── run_windows.bat
├── README.md
├── LICENSE
├── visioncloak/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── effects.py
│   └── segmentation.py
├── tests/
│   └── test_effects.py
└── output/
```

## How it works

1. OpenCV reads frames from the webcam.
2. MediaPipe Selfie Segmentation estimates a per-pixel human confidence mask.
3. The mask is cleaned with morphological operations and softened with Gaussian blur.
4. The application blends the live frame with the previously captured clean background:
   - Human mask: use the clean background.
   - Background mask: use the current live frame.
5. OpenCV displays the resulting frame in real time.

## Limitations and ways to improve the demo

- **Camera movement:** Moving the camera shifts the scene, so the saved background no longer aligns. Use a tripod or fixed laptop.
- **Changing background objects:** Objects that move after background capture will not be reconstructed correctly behind the person.
- **Lighting changes:** Large lighting changes can make the replacement visible.
- **Segmentation edges:** Loose hair, fast movement, and similar-coloured clothing/backgrounds can affect the mask.
- **Performance:** Real-time speed depends on CPU/GPU, camera resolution, and system load.

Potential future improvements include temporal mask smoothing, background alignment, person detection with a modern segmentation model, a polished desktop UI, and performance benchmarking.

## Testing

```bash
pytest -q
```

The unit test checks that the compositing function replaces masked pixels with the captured background while preserving unmasked pixels.

## Interview explanation

**Problem:** A webcam feed normally shows a person in front of the camera. This project creates an invisibility-style visual effect by replacing the detected person region with a clean frame of the empty scene.

**Approach:** I used MediaPipe Selfie Segmentation to generate a human mask, OpenCV to capture and process webcam frames, and NumPy alpha blending to composite the clean background into the human region.

**Key engineering decision:** The background is captured before the person enters. This avoids trying to invent unseen pixels and makes the effect understandable and reproducible.

**Evaluation ideas:** Measure average FPS, test segmentation quality under different lighting and clothing conditions, and compare the effect at different confidence thresholds.

**Honest description:** This is a real-time computer-vision prototype. It does not make a person physically invisible and does not guarantee perfect results in every environment.

## License

MIT License. See `LICENSE`.
