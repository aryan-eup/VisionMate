"""Text recognition (OCR) using Apple's Vision framework."""

import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import Foundation
    import objc
    import Vision
except ImportError as error:
    raise ImportError(
        "Apple Vision bindings are missing. Run: pip install pyobjc-framework-Vision"
    ) from error


class OCRError(RuntimeError):
    """Raised when text recognition fails."""


@dataclass(frozen=True)
class TextLine:
    text: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in pixels of the frame that was passed in


class TextReader:
    def __init__(self, min_confidence=0.5, languages=("en-US",), frame_is_mirrored=True):
        self.min_confidence = min_confidence
        self.languages = tuple(languages)
        # Camera frames are mirrored by default, which makes text unreadable.
        self.frame_is_mirrored = frame_is_mirrored

    def read(self, frame):
        """Recognise text in one BGR frame. Returns TextLine items in reading order."""
        if frame is None or frame.size == 0:
            raise OCRError("Empty frame passed to the text reader.")

        height, width = frame.shape[:2]
        image = cv2.flip(frame, 1) if self.frame_is_mirrored else frame

        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise OCRError("Could not encode the frame for text recognition.")
        data = Foundation.NSData.dataWithBytes_length_(encoded.tobytes(), len(encoded))

        lines = []
        with objc.autorelease_pool():
            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(True)
            request.setRecognitionLanguages_(list(self.languages))

            success, error = handler.performRequests_error_([request], None)
            if not success:
                raise OCRError(f"Apple Vision failed: {error}")

            for observation in request.results() or []:
                candidates = observation.topCandidates_(1)
                if not candidates:
                    continue

                text = str(candidates[0].string()).strip()
                confidence = float(candidates[0].confidence())
                if confidence < self.min_confidence or not _is_meaningful(text):
                    continue

                # Vision boxes are normalised (0-1) with the origin at the bottom-left.
                rect = observation.boundingBox()
                x1 = int(rect.origin.x * width)
                x2 = int((rect.origin.x + rect.size.width) * width)
                y1 = int((1 - rect.origin.y - rect.size.height) * height)
                y2 = int((1 - rect.origin.y) * height)
                lines.append(TextLine(text, confidence, (x1, y1, x2, y2)))

        # Reading order: top to bottom in rows, then left to right.
        lines.sort(key=lambda l: (round(((l.box[1] + l.box[3]) / 2) / height / 0.03), l.box[0]))

        if self.frame_is_mirrored:
            # Map boxes back onto the mirrored frame the caller is displaying.
            lines = [
                TextLine(l.text, l.confidence, (width - l.box[2], l.box[1], width - l.box[0], l.box[3]))
                for l in lines
            ]
        return lines


def _is_meaningful(text):
    return len(text) >= 2 and any(character.isalnum() for character in text)


def speakable_text(lines, max_chars=300):
    """Join recognised lines into one string that is comfortable to listen to."""
    text = " ".join(" ".join(line.text for line in lines).split())
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " and more."
    return text


def draw_text_boxes(frame, lines):
    """Return a copy of the frame with a box and the recognised text for each line."""
    annotated = frame.copy()
    color = (255, 200, 0)
    for line in lines:
        x1, y1, x2, y2 = line.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, line.text, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated


def _run_synthetic_checks():
    image = np.full((300, 900, 3), 255, dtype=np.uint8)
    cv2.putText(image, "HELLO VISIONMATE", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 5)
    cv2.putText(image, "OCR TEST 2026", (30, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 5)

    checks = (
        ("Synthetic image check", TextReader(frame_is_mirrored=False), image),
        ("Mirrored-frame check", TextReader(frame_is_mirrored=True), cv2.flip(image, 1)),
    )
    for name, reader, test_image in checks:
        lines = reader.read(test_image)
        text = " ".join(line.text for line in lines).upper()
        passed = "HELLO" in text and "2026" in text
        print(f"{name}: {'PASSED' if passed else 'FAILED'} (read: {text or 'nothing'})")


def _run_live_test():
    from src.camera import Camera
    from src.speech import Speaker

    reader = TextReader()
    speaker = Speaker()
    window_name = "VisionMate - OCR test (SPACE = read text, q = quit)"

    frozen = None
    show_until = 0.0

    try:
        with Camera(width=1280, height=720) as camera:
            first = camera.read()
            print(f"Camera running at {first.shape[1]}x{first.shape[0]}")

            while True:
                frame = camera.read()

                if frozen is not None and time.time() < show_until:
                    display = frozen
                else:
                    display = frame.copy()
                    cv2.putText(display, "SPACE = read text | q = quit", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == 32:  # space bar
                    start = time.perf_counter()
                    lines = reader.read(frame)
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    print(f"Read {len(lines)} line(s) in {elapsed_ms:.0f} ms")
                    for line in lines:
                        print(f"  [{line.confidence:.2f}] {line.text}")

                    frozen = draw_text_boxes(frame, lines)
                    show_until = time.time() + 3.0
                    speaker.speak(speakable_text(lines) or "No text found.", interrupt=True)
    finally:
        speaker.stop()
        cv2.destroyAllWindows()
    print("OCR test finished.")


if __name__ == "__main__":
    try:
        _run_synthetic_checks()
        _run_live_test()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)