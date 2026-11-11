
import os
import cv2
import gradio as gr

from visioncloak.segmentation import HumanSegmenter
from visioncloak.effects import apply_invisibility_effect

segmenter = HumanSegmenter(model_selection=1)


def capture_background(frame):
    if frame is None:
        return None, "Please enable your webcam first."

    background = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return background, "Background captured successfully!"


def process_frame(frame, background):
    if frame is None:
        return None

    if background is None:
        return frame

    live_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    mask = segmenter.get_mask(live_bgr, threshold=0.65)

    result = apply_invisibility_effect(
        live_bgr,
        background,
        mask
    )

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


with gr.Blocks(title="VisionCloak AI Human Invisibility") as demo:

    gr.Markdown("# VisionCloak – AI Human Invisibility")
    gr.Markdown("Capture an empty background before entering the camera view.")

    background_state = gr.State(None)

    camera = gr.Image(
        sources=["webcam"],
        type="numpy",
        streaming=True,
        label="Webcam"
    )

    capture_button = gr.Button("Capture Empty Background")
    status = gr.Textbox(label="Status")
    output = gr.Image(label="Invisibility Result")

    capture_button.click(
        capture_background,
        inputs=camera,
        outputs=[background_state, status]
    )

    camera.stream(
        process_frame,
        inputs=[camera, background_state],
        outputs=output,
        stream_every=0.2
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
