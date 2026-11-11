"""Gradio webcam app for VisionCloak."""
import os
import cv2
import gradio as gr

from visioncloak.segmentation import HumanSegmenter
from visioncloak.effects import apply_invisibility_effect

segmenter = HumanSegmenter(model_selection=1)


def capture_background(frame):
    if frame is None:
        return None, "No webcam frame received. Click Record on the webcam and wait for the live preview before capturing."

    background = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return background, "Empty background captured. Now enter the camera view."


def process_frame(frame, background):
    if frame is None:
        return None, None

    # Keep the most recent browser webcam frame in Gradio state.
    latest_frame = frame.copy()

    if background is None:
        return frame, latest_frame

    live_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    human_mask = segmenter.get_mask(live_bgr, threshold=0.65)
    result_bgr = apply_invisibility_effect(
        live_bgr, background, human_mask
    )
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return result_rgb, latest_frame


def capture_from_latest_frame(frame):
    return capture_background(frame)


def reset_background():
    return None, "Background reset. Keep the scene empty and capture again."


with gr.Blocks(title="VisionCloak AI Human Invisibility") as demo:
    gr.Markdown("# VisionCloak – AI Human Invisibility")
    gr.Markdown(
        "1. Allow browser webcam access and click **Record** on the webcam.\n"
        "2. Wait until the live preview appears. Keep the scene empty.\n"
        "3. Click **Capture Empty Background**.\n"
        "4. Enter the camera view. Keep the camera still and lighting steady."
    )

    background_state = gr.State(value=None)
    latest_frame_state = gr.State(value=None)

    camera = gr.Image(
        sources=["webcam"],
        type="numpy",
        streaming=True,
        label="Webcam – click Record to start"
    )

    with gr.Row():
        capture_button = gr.Button(
            "Capture Empty Background", variant="primary"
        )
        reset_button = gr.Button("Reset Background")

    status = gr.Textbox(
        label="Status",
        value="Allow webcam access, click Record, and wait for a live frame.",
        interactive=False
    )
    output = gr.Image(label="VisionCloak Result")

    capture_button.click(
        fn=capture_from_latest_frame,
        inputs=latest_frame_state,
        outputs=[background_state, status]
    )

    reset_button.click(
        fn=reset_background,
        inputs=None,
        outputs=[background_state, status]
    )

    camera.stream(
        fn=process_frame,
        inputs=[camera, background_state],
        outputs=[output, latest_frame_state],
        stream_every=0.2,
        concurrency_limit=1
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
