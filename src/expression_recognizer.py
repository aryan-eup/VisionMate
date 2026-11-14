"""Face detection + expression classification with temporal smoothing."""

import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from src.expression_model import IMAGE_SIZE, load_checkpoint, pick_device, preprocess
from src.position import describe_position, get_position

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "models" / "expression_cnn.pt"
DATA_FILE = PROJECT_ROOT / "data" / "fer2013_processed.npz"


class RecognizerError(RuntimeError):
    """Raised when the expression recognizer cannot be set up."""


@dataclass(frozen=True)
class ExpressionResult:
    label: str
    confidence: float      # smoothed probability of the label
    confident: bool        # False when below min_confidence: do not announce
    box: tuple             # (x1, y1, x2, y2) of the face in frame pixels
    position: str          # left / centre / right
    probabilities: dict    # smoothed probability per class


class ProbabilitySmoother:
    """Average the last few probability vectors; forget them after a gap."""

    def __init__(self, window=5, reset_after_misses=3):
        self._history = deque(maxlen=window)
        self._misses = 0
        self._reset_after_misses = reset_after_misses

    def update(self, probabilities):
        self._misses = 0
        self._history.append(np.asarray(probabilities, dtype=np.float64))
        return np.mean(self._history, axis=0)

    def miss(self):
        self._misses += 1
        if self._misses >= self._reset_after_misses:
            self._history.clear()

    def reset(self):
        self._history.clear()
        self._misses = 0


class ExpressionRecognizer:
    def __init__(self, model_path=MODEL_FILE, min_confidence=0.45,
                 smoothing_window=5, min_face_size=60, device=None):
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self.device = device or pick_device()
        self.model, self.class_names, _ = load_checkpoint(model_path, self.device)

        cascade_file = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_file))
        if self._cascade.empty():
            raise RecognizerError(f"Could not load the face detector: {cascade_file}")

        self._smoother = ProbabilitySmoother(window=smoothing_window)
        self._classify(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))  # warm-up

    def reset(self):
        self._smoother.reset()

    @torch.no_grad()
    def _classify(self, face):
        """Class probabilities for one 48x48 uint8 grayscale face."""
        tensor = preprocess(face).to(self.device)
        return torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()

    def analyze(self, frame):
        """Return an ExpressionResult for the largest face in the frame, or None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        if len(faces) == 0:
            self._smoother.miss()
            return None

        x, y, w, h = (int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
        frame_height, frame_width = gray.shape

        pad = int(0.1 * w)  # a little context around the detected box
        crop = gray[max(y - pad, 0):min(y + h + pad, frame_height),
                    max(x - pad, 0):min(x + w + pad, frame_width)]
        face = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        probabilities = self._smoother.update(self._classify(face))
        best = int(probabilities.argmax())
        box = (x, y, x + w, y + h)

        return ExpressionResult(
            label=self.class_names[best],
            confidence=float(probabilities[best]),
            confident=bool(probabilities[best] >= self.min_confidence),
            box=box,
            position=get_position(box, frame_width),
            probabilities={name: float(p) for name, p in zip(self.class_names, probabilities)},
        )


def describe_expression(result):
    """Sentence to speak, e.g. 'Face detected on the left. Visible expression: happy.'"""
    return f"{describe_position('face', result.position)} Visible expression: {result.label}."


def draw_expression(frame, result):
    """Return a copy of the frame with the face box, label and probability bars."""
    annotated = frame.copy()
    if result is None:
        return annotated

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 200, 0) if result.confident else (160, 160, 160)
    x1, y1, x2, y2 = result.box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

    text = (f"{result.label} {result.confidence:.0%}" if result.confident
            else f"unsure ({result.label} {result.confidence:.0%})")
    cv2.putText(annotated, text, (x1, max(y1 - 8, 18)), font, 0.7, color, 2)

    panel_x, panel_y, row_height = annotated.shape[1] - 175, 10, 20
    cv2.rectangle(annotated, (panel_x - 6, panel_y - 6),
                  (panel_x + 169, panel_y + row_height * len(result.probabilities) + 2),
                  (0, 0, 0), -1)
    for index, (name, probability) in enumerate(result.probabilities.items()):
        y = panel_y + index * row_height
        bar_color = (0, 220, 255) if name == result.label else (200, 200, 200)
        cv2.rectangle(annotated, (panel_x + 75, y),
                      (panel_x + 75 + int(probability * 90), y + 12), bar_color, -1)
        cv2.putText(annotated, name, (panel_x, y + 11), font, 0.4, (255, 255, 255), 1)
    return annotated


# ---------------------------------------------------------------- tests

def _run_checks():
    # 1. Smoother behaviour.
    smoother = ProbabilitySmoother(window=5, reset_after_misses=3)
    for _ in range(3):
        smoother.update([0.8, 0.2])
    assert smoother.update([0.1, 0.9]).argmax() == 0, "one outlier should not flip the label"

    smoother = ProbabilitySmoother(window=2, reset_after_misses=3)
    smoother.update([1, 0])
    smoother.update([1, 0])
    assert np.allclose(smoother.update([0, 1]), [0.5, 0.5]), "window should keep only the last 2"

    smoother = ProbabilitySmoother(window=5, reset_after_misses=3)
    smoother.update([1, 0])
    smoother.miss()
    smoother.miss()
    assert np.allclose(smoother.update([0, 1]), [0.5, 0.5]), "two misses should keep history"
    for _ in range(3):
        smoother.miss()
    assert np.allclose(smoother.update([0, 1]), [0, 1]), "three misses should clear history"
    print("Check 1 passed: smoothing logic")

    # 2. Model and detector load; a blank frame has no face.
    recognizer = ExpressionRecognizer()
    assert len(recognizer.class_names) == 7
    assert recognizer.analyze(np.zeros((480, 640, 3), dtype=np.uint8)) is None
    print(f"Check 2 passed: model loaded on {recognizer.device}, blank frame gives no face")

    # 3. The recognizer's own classification path reproduces training accuracy.
    if not DATA_FILE.exists():
        print("Check 3 skipped: processed dataset not found")
        return
    data = np.load(DATA_FILE, allow_pickle=False)
    assert [str(n) for n in data["class_names"]] == recognizer.class_names
    x_test, y_test = data["x_test"], data["y_test"]
    chosen = np.random.default_rng(0).choice(len(y_test), size=400, replace=False)
    correct = sum(int(recognizer._classify(x_test[i]).argmax() == y_test[i]) for i in chosen)
    accuracy = correct / len(chosen)
    assert accuracy > 0.5, f"accuracy {accuracy:.2f} is too low: preprocessing mismatch?"
    print(f"Check 3 passed: {accuracy:.1%} accuracy on 400 random test faces")


def _run_live_test():
    from src.camera import Camera

    recognizer = ExpressionRecognizer()
    window_name = "VisionMate - expression test (press q to quit)"
    interval = 0.1  # seconds between analyses
    last_run, analysis_ms, result, last_key = 0.0, 0.0, None, "start"

    with Camera() as camera:
        while True:
            frame = camera.read()

            now = time.monotonic()
            if now - last_run >= interval:
                last_run = now
                start = time.perf_counter()
                result = recognizer.analyze(frame)
                analysis_ms = (time.perf_counter() - start) * 1000

                key = result.label if result and result.confident else None
                if key != last_key:
                    last_key = key
                    if result is None:
                        print("no face")
                    else:
                        print(describe_expression(result) if result.confident
                              else f"unsure ({result.label} {result.confidence:.0%})")

            annotated = draw_expression(frame, result)
            cv2.putText(annotated, f"{analysis_ms:.0f} ms", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("Expression test finished.")


if __name__ == "__main__":
    try:
        _run_checks()
        _run_live_test()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)