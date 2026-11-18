"""VisionMate - Streamlit interface."""

import time

import cv2
import streamlit as st

from src import config
from src.camera import Camera, CameraError
from src.pipeline import VisionPipeline

FEATURES = (
    ("near", "Near warnings"),
    ("objects", "Object detection + position"),
    ("text", "Text reading (OCR)"),
    ("expression", "Facial expression"),
)
DEFAULT_ON = {"near"}   # safety feature: on as soon as the page opens
MAX_LOG_LINES = 10

st.set_page_config(page_title="VisionMate", page_icon="👁️", layout="wide")


@st.cache_resource(show_spinner="Loading models, this takes a few seconds...")
def load_pipeline():
    return VisionPipeline()


def init_state():
    defaults = {"running": False, "camera": None, "log": [],
                "force_off": [], "notice": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    for name, _ in FEATURES:
        st.session_state.setdefault(f"{name}_on", name in DEFAULT_ON)

    # Features the pipeline switched off after an error: untick their boxes.
    # This must happen before the checkboxes are created.
    for name in st.session_state.force_off:
        st.session_state[f"{name}_on"] = False
    st.session_state.force_off = []


def add_log(text):
    log = st.session_state.log
    log.insert(0, f"{time.strftime('%H:%M:%S')}  {text}")
    del log[MAX_LOG_LINES:]


def render_log(placeholder):
    lines = st.session_state.log
    placeholder.text("\n\n".join(lines) if lines else "Nothing spoken yet.")


def start_camera(pipeline):
    camera = Camera(index=config.CAMERA_INDEX, width=config.CAMERA_WIDTH,
                    height=config.CAMERA_HEIGHT, mirror=config.CAMERA_MIRROR)
    try:
        camera.start()
    except CameraError as error:
        st.session_state.notice = ("error", str(error))
        return
    pipeline.reset()
    st.session_state.camera = camera
    st.session_state.running = True


def stop_camera(pipeline):
    pipeline.stop_speech()
    if st.session_state.camera is not None:
        st.session_state.camera.stop()
    st.session_state.camera = None
    st.session_state.running = False


def run_live_view(pipeline, video, status, log_box):
    camera = st.session_state.camera
    fps, last_time, frames = 0.0, time.time(), 0

    while st.session_state.running:
        try:
            frame = camera.read()
        except CameraError as error:
            stop_camera(pipeline)
            st.session_state.notice = ("error", f"Camera stopped: {error}")
            st.rerun()

        result = pipeline.process(frame)
        video.image(cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB),
                    channels="RGB", output_format="JPEG")

        for event in result.events:
            add_log(event)
        if result.events:
            render_log(log_box)

        # A feature that is ticked but switched off by the pipeline has crashed.
        failed = [name for name, _ in FEATURES
                  if st.session_state[f"{name}_on"]
                  and not getattr(pipeline, f"{name}_enabled")]
        if failed:
            warnings = [e for e in result.events if e.startswith("Warning")]
            st.session_state.force_off = failed
            st.session_state.notice = (
                "warning", " ".join(warnings) or "A feature was switched off after an error.")
            st.rerun()

        now = time.time()
        instant = 1.0 / max(now - last_time, 1e-6)
        fps = 0.9 * fps + 0.1 * instant if fps else instant
        last_time = now
        frames += 1
        if frames % 10 == 0:
            timings = " | ".join(f"{name} {ms:.0f} ms" for name, ms in pipeline.stats.items())
            status.caption(f"{fps:.0f} FPS" + (f" | {timings}" if timings else "")
                           + (" | speaking" if pipeline.speaker.is_speaking else ""))


# ------------------------------------------------------------------ page

init_state()
pipeline = load_pipeline()

st.title("VisionMate")
st.caption("See. Warn. Call for help. An AI visual assistant that watches through "
           "the camera and speaks what it sees.")

for warning in pipeline.warnings:
    st.warning(warning)

notice = st.session_state.notice
if notice:
    getattr(st, notice[0])(notice[1])
    st.session_state.notice = None

with st.sidebar:
    st.header("Controls")
    if st.session_state.running:
        st.button("Stop camera", on_click=stop_camera, args=(pipeline,), type="primary")
    else:
        st.button("Start camera", on_click=start_camera, args=(pipeline,), type="primary")

    st.subheader("Features")
    st.checkbox("Near warnings", key="near_on",
                help="Warns when an obstacle looks close. This is an approximation "
                     "from how large the object appears, not true depth.")
    st.checkbox("Object detection + position", key="objects_on",
                help="Announces what it sees and whether it is on the left, "
                     "in the centre or on the right.")
    st.checkbox("Text reading (OCR)", key="text_on",
                help="Reads printed text it sees. Use 'Read text now' to read on demand.")
    st.checkbox("Facial expression", key="expression_on",
                disabled=not pipeline.expression_available,
                help="Reports the predicted visible facial expression.")

    st.subheader("Speech")
    read_now = st.button("Read text now", disabled=not st.session_state.running)
    stop_speech = st.button("Stop speech")

    st.divider()
    st.caption(
        "VisionMate is a demonstration project, not a safety device. "
        "Expressions are predictions of a visible facial expression, "
        "not a person's actual feelings."
    )

pipeline.set_enabled(
    near=st.session_state.near_on,
    objects=st.session_state.objects_on,
    text=st.session_state.text_on,
    expression=st.session_state.expression_on,
)

if stop_speech:
    pipeline.stop_speech()

if read_now and st.session_state.running:
    try:
        add_log(pipeline.read_text_now(st.session_state.camera.read()))
    except Exception as error:
        st.session_state.notice = ("error", f"Could not read text: {error}")
        st.rerun()

video_column, log_column = st.columns([3, 1])
with video_column:
    video = st.empty()
    status = st.empty()
with log_column:
    st.subheader("Spoken")
    log_box = st.empty()
render_log(log_box)

if st.session_state.running:
    run_live_view(pipeline, video, status, log_box)
else:
    video.info("The camera is off. Press 'Start camera' in the sidebar.")