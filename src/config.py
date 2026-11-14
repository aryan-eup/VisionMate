"""Central place for every tunable number in VisionMate."""

# --- Camera -------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280          # full-resolution frame, used for OCR
CAMERA_HEIGHT = 720
CAMERA_MIRROR = True         # True = preview behaves like a mirror (laptop demo)
PROCESS_WIDTH = 640          # frames are shrunk to this width for YOLO and faces

# --- Speech -------------------------------------------------------------
SPEECH_RATE = 170            # words per minute
SPEECH_VOICE = None          # None = system default; e.g. "Samantha"

# --- Object detection ---------------------------------------------------
YOLO_MODEL = "yolo11n.pt"
YOLO_CONFIDENCE = 0.45
OBJECT_INTERVAL = 0.4        # seconds between YOLO runs
OBJECT_COOLDOWN = 8.0        # seconds before the same object/zone is repeated
OBJECT_MIN_CONSECUTIVE = 2   # runs in a row before an object is announced
MAX_OBJECTS_PER_MESSAGE = 3

# --- Text recognition (OCR) ---------------------------------------------
OCR_INTERVAL = 3.0           # seconds between automatic OCR runs
OCR_COOLDOWN = 25.0          # seconds before similar text is read again
OCR_MIN_CONFIDENCE = 0.5
OCR_MAX_CHARS = 300          # longest text read aloud in one go
OCR_SIMILARITY = 0.8         # 0-1: how alike two readings must be to count as "the same"

# --- Facial expression --------------------------------------------------
EXPRESSION_INTERVAL = 0.2    # seconds between face analyses
EXPRESSION_MIN_CONFIDENCE = 0.45
EXPRESSION_STABLE_RUNS = 6   # confident analyses in a row before announcing
EXPRESSION_COOLDOWN = 15.0   # seconds before the same expression is repeated


def settings():
    """All settings as a {NAME: value} dictionary."""
    return {name: value for name, value in globals().items() if name.isupper()}


def validate():
    """Raise ValueError if any setting is obviously wrong."""
    current = settings()

    for name, value in current.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
            raise ValueError(f"{name} must not be negative (got {value})")

    for name in ("YOLO_CONFIDENCE", "EXPRESSION_MIN_CONFIDENCE",
                 "OCR_MIN_CONFIDENCE", "OCR_SIMILARITY"):
        if not 0 < current[name] <= 1:
            raise ValueError(f"{name} must be between 0 and 1 (got {current[name]})")

    for name in ("OBJECT_INTERVAL", "OCR_INTERVAL", "EXPRESSION_INTERVAL",
                 "CAMERA_WIDTH", "CAMERA_HEIGHT", "PROCESS_WIDTH", "SPEECH_RATE",
                 "OBJECT_MIN_CONSECUTIVE", "MAX_OBJECTS_PER_MESSAGE",
                 "EXPRESSION_STABLE_RUNS"):
        if current[name] <= 0:
            raise ValueError(f"{name} must be positive (got {current[name]})")

    if current["CAMERA_WIDTH"] < current["PROCESS_WIDTH"]:
        raise ValueError("CAMERA_WIDTH must be at least PROCESS_WIDTH")


if __name__ == "__main__":
    for name, value in settings().items():
        print(f"{name:<28}{value}")
    validate()
    print("\nConfig OK.")