"""Convert bounding boxes into left / centre / right positions."""

import sys
import time
from dataclasses import dataclass

import cv2

LEFT, CENTRE, RIGHT = "left", "centre", "right"

_PHRASES = {
    LEFT: "on the left",
    CENTRE: "in the centre",
    RIGHT: "on the right",
}


def get_position(box, frame_width):
    """Return LEFT, CENTRE or RIGHT for a (x1, y1, x2, y2) box."""
    if frame_width <= 0:
        raise ValueError("frame_width must be positive")

    x1, _, x2, _ = box
    centre_ratio = ((x1 + x2) / 2) / frame_width

    if centre_ratio < 1 / 3:
        return LEFT
    if centre_ratio > 2 / 3:
        return RIGHT
    return CENTRE


def describe_position(label, position):
    """Build a sentence such as 'Person detected on the left.'"""
    return f"{label.capitalize()} detected {_PHRASES[position]}."


@dataclass(frozen=True)
class PositionedDetection:
    detection: object  # a Detection from object_detector.py
    position: str

    @property
    def label(self):
        return self.detection.label

    @property
    def sentence(self):
        return describe_position(self.label, self.position)


def locate(detections, frame_width):
    """Attach a position to every detection."""
    return [
        PositionedDetection(d, get_position(d.box, frame_width))
        for d in detections
    ]


def _run_unit_checks():
    width = 640  # thirds are at about 213 and 427
    checks = [
        ((0, 0, 100, 100), LEFT),
        ((200, 0, 220, 10), LEFT),      # centre x = 210
        ((210, 0, 230, 10), CENTRE),    # centre x = 220
        ((270, 0, 370, 100), CENTRE),
        ((420, 0, 430, 10), CENTRE),    # centre x = 425
        ((430, 0, 440, 10), RIGHT),     # centre x = 435
        ((500, 0, 600, 100), RIGHT),
        ((0, 0, 640, 480), CENTRE),     # object filling the whole frame
    ]
    for box, expected in checks:
        actual = get_position(box, width)
        assert actual == expected, f"{box}: expected {expected}, got {actual}"

    assert describe_position("person", LEFT) == "Person detected on the left."
    assert describe_position("chair", CENTRE) == "Chair detected in the centre."
    assert describe_position("dog", RIGHT) == "Dog detected on the right."

    print(f"Unit checks passed ({len(checks) + 3} checks).")


def _run_live_test():
    from src.camera import Camera
    from src.object_detector import ObjectDetector, draw_detections

    print("Loading model...")
    detector = ObjectDetector()
    window_name = "VisionMate - position test (press q to quit)"
    last_print = 0.0

    with Camera() as camera:
        while True:
            frame = camera.read()
            height, width = frame.shape[:2]

            detections = detector.detect(frame)
            positioned = locate(detections, width)

            annotated = draw_detections(frame, detections)
            for x in (width // 3, 2 * width // 3):
                cv2.line(annotated, (x, 0), (x, height), (255, 255, 255), 1)
            for name, x in ((LEFT, width // 6), (CENTRE, width // 2), (RIGHT, 5 * width // 6)):
                cv2.putText(annotated, name.upper(), (x - 30, height - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for item in positioned:
                x1, _, _, y2 = item.detection.box
                cv2.putText(annotated, item.position, (x1 + 4, y2 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(window_name, annotated)

            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                if positioned:
                    print(" | ".join(item.sentence for item in positioned))
                else:
                    print("nothing detected")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("Position test finished.")


if __name__ == "__main__":
    try:
        _run_unit_checks()
        _run_live_test()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)