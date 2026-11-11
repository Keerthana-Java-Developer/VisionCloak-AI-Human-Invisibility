# VisionCloak — AI-Powered Human Invisibility

<p align="center">
  <strong>A real-time computer-vision project using Python, OpenCV and MediaPipe.</strong>
  <br />
  Detect a person in a webcam stream and replace the segmented region with a clean background frame captured earlier.
</p>

<p align="center">
  <a href="https://visioncloak-ai-human-invisibility.onrender.com">Live Demo</a> ·
  <a href="https://github.com/keerthanavinoth2003-maker/VisionCloak-AI-Human-Invisibility">Source Code</a>
</p>

> **Project note:** This is an experimental visual effect, not physical invisibility. Results depend on a fixed camera, stable lighting, and a clean background captured before the person enters the scene. Shadow removal is experimental and may not work perfectly in every environment.

## Features

- Real-time webcam processing with OpenCV
- Human segmentation with MediaPipe Selfie Segmentation
- Mask cleanup and softened compositing edges
- Background capture, reset, and replacement
- Snapshot saving and MP4 recording
- Configurable camera index, resolution, and segmentation threshold
- Separate modules for application logic, configuration, segmentation, and effects
- A basic automated test for the compositing function
- Browser-based webcam demo powered by Gradio

## Live demo

**[Open VisionCloak on Render](https://visioncloak-ai-human-invisibility.onrender.com)**

The browser demo needs webcam permission. Start the webcam preview, capture the empty scene, and then enter the frame. The hosted service may take a short time to wake up.

## How it works

1. OpenCV captures frames from the webcam.
2. MediaPipe Selfie Segmentation estimates which pixels belong to a person.
3. The application cleans and expands the mask to cover the detected person.
4. The detected region is blended with the previously captured empty-scene background.
5. OpenCV or Gradio displays the processed frame.

## Demo workflow

1. Keep the camera still and make sure the scene is empty.
2. Capture the clean background.
3. Enter the camera view.
4. Enable the invisibility effect.
5. Reset and capture again if the camera position or lighting changes.

## Tech stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| OpenCV | Webcam capture, image processing, and display |
| MediaPipe | Human segmentation |
| NumPy | Mask operations and image compositing |
| Gradio | Browser-based webcam interface |
| pytest | Basic automated testing |

## Project structure

```text
VisionCloak-AI-Human-Invisibility/
├── main.py
├── web_app.py
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

## Run locally

### Requirements

- Python 3.10 or 3.11 (64-bit recommended)
- A webcam
- Windows, macOS, or Linux

### Windows PowerShell

Run these commands from the repository root:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks virtual-environment activation, you can use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

### Run the browser app locally

```bash
python web_app.py
```

Then open the local Gradio URL printed in the terminal and allow webcam access.

## Desktop keyboard controls

| Key | Action |
|---|---|
| `B` | Capture the empty background |
| `I` | Toggle the invisibility effect |
| `R` | Reset the captured background |
| `S` | Save a snapshot in `output/` |
| `V` | Start or stop MP4 recording |
| `Q` | Quit the application |

### Optional settings

```bash
python main.py --camera 0 --width 1280 --height 720 --threshold 0.65
```

- `--camera`: webcam device index; try `1` if camera `0` is unavailable.
- `--width` and `--height`: requested camera resolution.
- `--threshold`: segmentation confidence threshold between `0.0` and `1.0`.
- `--no-mirror`: disable the mirrored preview.
- `--output-dir`: choose another output folder.
- `--debug`: enable detailed logs.

## Run tests

```bash
pytest -q
```

## Known limitations

- Camera movement causes the captured background to become misaligned.
- Lighting changes can make the replacement region visible.
- Moving objects in the background are not reconstructed.
- Hair, fast movement, and similar clothing/background colours can reduce segmentation quality.
- Cast-shadow removal is experimental; some shadows may remain or other dark regions may be mistakenly selected.
- Performance depends on the computer, webcam resolution, and available processing power.

For best results, use a fixed camera, even front lighting, and capture the background with nobody in the scene.

## Future improvements

- Temporal mask smoothing to reduce flicker
- Better background alignment and shadow estimation
- Performance profiling and FPS display
- More comprehensive automated tests
- Improved browser UI and clearer setup diagnostics

## What I learned

This project explores real-time computer vision, human segmentation, image masks, morphological operations, alpha blending, webcam streaming, and deployment of a Python web application.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
