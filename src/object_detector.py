"""YOLO object detection wrapper."""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class DetectorError(RuntimeError):
    """Raised when the detection model cannot be loaded."""


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in pixels


class ObjectDetector:
    def __init__(self, model_name="yolo11n.pt", confidence=0.45,
                 image_size=640, device=None):
        self.confidence = confidence
        self.image_size = image_size
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        MODELS_DIR.mkdir(exist_ok=True)
        try:
            # Downloads the pretrained weights on first use.
            self.model = YOLO(str(MODELS_DIR / model_name))
        except Exception as error:
            raise DetectorError(
                f"Could not load '{model_name}'. The first run needs an "
                f"internet connection to download the weights. ({error})"
            ) from error

        # The first inference is slow, so do it once up front.
        self.detect(np.zeros((image_size, image_size, 3), dtype=np.uint8))

    def detect(self, frame):
        """Run YOLO on one BGR frame. Returns detections, most confident first."""
        result = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    label=result.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    box=(x1, y1, x2, y2),
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


def draw_detections(frame, detections):
    """Return a copy of the frame with boxes, names and confidence scores."""
    annotated = frame.copy()
    color = (0, 200, 0)

    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{detection.label} {detection.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(annotated, (x1, top),
                      (x1 + text_w + 6, top + text_h + baseline + 4), color, -1)
        cv2.putText(annotated, text, (x1 + 3, top + text_h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return annotated


def _run_detector_test():
    from src.camera import Camera  # imported here, only needed for the test

    print("Loading model (first run downloads the weights)...")
    detector = ObjectDetector()
    print(f"Model loaded on device: {detector.device}")

    window_name = "VisionMate - object detection test (press q to quit)"
    last_print = 0.0

    with Camera() as camera:
        while True:
            frame = camera.read()

            start = time.perf_counter()
            detections = detector.detect(frame)
            inference_ms = (time.perf_counter() - start) * 1000

            annotated = draw_detections(frame, detections)
            cv2.putText(annotated,
                        f"{inference_ms:.0f} ms | {len(detections)} objects",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(window_name, annotated)

            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                summary = ", ".join(f"{d.label} ({d.confidence:.2f})" for d in detections)
                print(f"[{inference_ms:4.0f} ms] {summary or 'nothing detected'}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("Detection test finished.")


if __name__ == "__main__":
    try:
        _run_detector_test()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)