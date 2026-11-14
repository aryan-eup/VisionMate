"""Temporary diagnostic: see exactly what the expression model gets from the webcam.

Keys: e = cycle contrast mode | p = cycle crop padding | SPACE = print probabilities
      s = save the current face crop | q = quit
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera
from src.expression_model import IMAGE_SIZE
from src.expression_recognizer import DATA_FILE, ExpressionRecognizer, ProbabilitySmoother

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEBUG_DIR = PROJECT_ROOT / "assets" / "debug"
CASCADE_FILE = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
MODES = ("none", "equalize", "clahe")
PADDINGS = (0.0, 0.1, 0.2, 0.3)
PREVIEW_SIZE = 288
MODEL_WINDOW = "What the model sees (48x48, enlarged)"
MAIN_WINDOW = "VisionMate - webcam check (e, p, SPACE, s, q)"
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
FONT = cv2.FONT_HERSHEY_SIMPLEX


def apply_mode(face, mode):
    if mode == "equalize":
        return cv2.equalizeHist(face)
    if mode == "clahe":
        return CLAHE.apply(face)
    return face


def training_stats():
    """Typical brightness and contrast of the FER-2013 training faces."""
    if not DATA_FILE.exists():
        return None
    faces = np.load(DATA_FILE, allow_pickle=False)["x_train"][::5].astype(np.float32)
    flat = faces.reshape(len(faces), -1)
    return float(flat.mean(axis=1).mean()), float(flat.std(axis=1).mean())


def crop_face(gray, box, padding):
    x, y, w, h = box
    pad = int(padding * w)
    height, width = gray.shape
    crop = gray[max(y - pad, 0):min(y + h + pad, height),
                max(x - pad, 0):min(x + w + pad, width)]
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def enlarge(face):
    return cv2.resize(face, (PREVIEW_SIZE, PREVIEW_SIZE), interpolation=cv2.INTER_NEAREST)


def draw_bars(image, probabilities, class_names):
    best = int(np.argmax(probabilities))
    cv2.rectangle(image, (6, 40), (190, 48 + 20 * len(class_names)), (0, 0, 0), -1)
    for index, (name, probability) in enumerate(zip(class_names, probabilities)):
        y = 50 + index * 20
        color = (0, 220, 255) if index == best else (200, 200, 200)
        cv2.putText(image, name, (12, y + 11), FONT, 0.42, (255, 255, 255), 1)
        cv2.rectangle(image, (85, y), (85 + int(probability * 95), y + 12), color, -1)


def main():
    recognizer = ExpressionRecognizer()
    cascade = cv2.CascadeClassifier(str(CASCADE_FILE))
    if cascade.empty():
        raise RuntimeError(f"Could not load the face detector: {CASCADE_FILE}")

    stats = training_stats()
    if stats:
        print(f"FER-2013 training faces: brightness {stats[0]:.0f}, contrast {stats[1]:.0f}")
    print("Keys: e = contrast mode | p = padding | SPACE = print probabilities "
          "| s = save crop | q = quit")

    smoother = ProbabilitySmoother(window=4)
    mode_index = padding_index = 0
    probabilities = face = model_input = None

    with Camera() as camera:
        while True:
            frame = camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                             minSize=(60, 60))
            mode, padding = MODES[mode_index], PADDINGS[padding_index]
            display = frame.copy()

            if len(faces):
                box = tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
                face = crop_face(gray, box, padding)
                model_input = apply_mode(face, mode)
                probabilities = smoother.update(recognizer._classify(model_input))

                x, y, w, h = box
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 0), 2)
                draw_bars(display, probabilities, recognizer.class_names)
                cv2.imshow(MODEL_WINDOW, enlarge(model_input))
            else:
                smoother.miss()
                probabilities = face = None

            info = f"mode={mode} pad={padding}"
            if face is not None:
                info += f" | crop brightness {face.mean():.0f} contrast {face.std():.0f}"
            else:
                info += " | no face"
            cv2.putText(display, info, (10, 25), FONT, 0.55, (0, 255, 0), 2)
            cv2.imshow(MAIN_WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("e"):
                mode_index = (mode_index + 1) % len(MODES)
                smoother.reset()
            elif key == ord("p"):
                padding_index = (padding_index + 1) % len(PADDINGS)
                smoother.reset()
            elif key == ord(" ") and probabilities is not None:
                ranked = sorted(zip(recognizer.class_names, probabilities),
                                key=lambda item: -item[1])
                print(f"[mode={mode} pad={padding} | brightness {face.mean():.0f} "
                      f"contrast {face.std():.0f}] "
                      + ", ".join(f"{name} {p:.2f}" for name, p in ranked))
            elif key == ord("s") and model_input is not None:
                DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                path = DEBUG_DIR / f"crop_{mode}_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(str(path), enlarge(model_input))
                print(f"Saved {path.relative_to(PROJECT_ROOT)}")

    cv2.destroyAllWindows()
    print("Webcam check finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)