"""Runs every VisionMate feature together on a stream of camera frames."""

import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

import cv2

from src import config
from src.announcer import Announcer
from src.expression_recognizer import (
    ExpressionRecognizer, describe_expression, draw_expression,
)
from src.object_detector import ObjectDetector, draw_detections
from src.position import locate
from src.proximity import VERY_CLOSE, HazardWarner, find_hazards
from src.speech import Speaker
from src.text_reader import TextLine, TextReader, draw_text_boxes, speakable_text

FACE_GONE_RUNS = 15           # analyses without a face before the face is "forgotten"
ON_DEMAND_BOX_SECONDS = 4.0   # how long boxes from "read text now" stay on screen
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class FrameResult:
    frame: object   # annotated BGR frame, PROCESS_WIDTH pixels wide
    events: list    # sentences spoken this frame, plus any "Warning: ..." messages


class VisionPipeline:
    def __init__(self, speaker=None):
        config.validate()
        self.speaker = speaker or Speaker(rate=config.SPEECH_RATE, voice=config.SPEECH_VOICE)
        self.announcer = Announcer(
            self.speaker,
            object_cooldown=config.OBJECT_COOLDOWN,
            min_consecutive=config.OBJECT_MIN_CONSECUTIVE,
            max_objects_per_message=config.MAX_OBJECTS_PER_MESSAGE,
        )
        self.hazard_warner = HazardWarner(
            self.speaker,
            cooldown=config.PROXIMITY_COOLDOWN,
            min_consecutive=config.PROXIMITY_MIN_CONSECUTIVE,
        )
        self.detector = ObjectDetector(model_name=config.YOLO_MODEL,
                                       confidence=config.YOLO_CONFIDENCE)
        self.reader = TextReader(min_confidence=config.OCR_MIN_CONFIDENCE,
                                 frame_is_mirrored=config.CAMERA_MIRROR)

        self.warnings = []
        try:
            self.recognizer = ExpressionRecognizer(
                min_confidence=config.EXPRESSION_MIN_CONFIDENCE)
        except Exception as error:
            self.recognizer = None
            self.warnings.append(f"Facial expression recognition is unavailable: {error}")

        self.objects_enabled = False
        self.near_enabled = False
        self.text_enabled = False
        self.expression_enabled = False
        self.stats = {}   # last run time of each feature, in milliseconds

        self._last_run = {"objects": 0.0, "text": 0.0, "expression": 0.0}
        self._detections, self._positioned, self._hazards = [], [], []
        self._text_lines, self._text_lines_until = [], 0.0
        self._spoken_texts = []   # (normalised text, time) of recently read text
        self._expression = None
        self._reset_expression_state()

    # ------------------------------------------------------------ control

    @property
    def expression_available(self):
        return self.recognizer is not None

    def set_enabled(self, objects=None, text=None, expression=None, near=None):
        """Switch features on or off. Arguments left as None are unchanged."""
        detection_was_on = self.objects_enabled or self.near_enabled

        if objects is not None and objects != self.objects_enabled:
            self.objects_enabled = objects
            if not objects:
                self.announcer.announce_objects([])   # forget "seen N times in a row"

        if near is not None and near != self.near_enabled:
            self.near_enabled = near
            if not near:
                self._hazards = []
                self.hazard_warner.reset()

        detection_is_on = self.objects_enabled or self.near_enabled
        if detection_is_on and not detection_was_on:
            self._last_run["objects"] = 0.0
        if not detection_is_on:
            self._detections, self._positioned = [], []

        if text is not None and text != self.text_enabled:
            self.text_enabled = text
            self._last_run["text"] = 0.0

        if (expression is not None and self.expression_available
                and expression != self.expression_enabled):
            self.expression_enabled = expression
            self._last_run["expression"] = 0.0
            if not expression:
                self._expression = None
                self._reset_expression_state()
                self.recognizer.reset()

    def reset(self):
        """Forget all history, e.g. when the camera is restarted."""
        self.speaker.stop()
        self.announcer.reset()
        self.hazard_warner.reset()
        self._detections, self._positioned, self._hazards = [], [], []
        self._text_lines, self._text_lines_until = [], 0.0
        self._spoken_texts = []
        self._expression = None
        self._reset_expression_state()
        if self.recognizer:
            self.recognizer.reset()
        for feature in self._last_run:
            self._last_run[feature] = 0.0

    def stop_speech(self):
        self.speaker.stop()

    def close(self):
        self.speaker.stop()

    def _reset_expression_state(self):
        self._stable_label = None
        self._stable_runs = 0
        self._announced_expression = None
        self._face_missing = 0

    def _disable(self, name):
        if name == "objects":   # detection feeds both of these
            self.objects_enabled = False
            self.near_enabled = False
        else:
            setattr(self, f"{name}_enabled", False)

    # ---------------------------------------------------------- processing

    def process(self, frame):
        """Handle one full-resolution camera frame."""
        now = time.monotonic()
        events = []
        small, scale = self._shrink(frame)

        schedule = (
            ("objects", self.objects_enabled or self.near_enabled,
             config.OBJECT_INTERVAL, self._update_objects),
            ("text", self.text_enabled, config.OCR_INTERVAL, self._update_text),
            ("expression", self.expression_enabled, config.EXPRESSION_INTERVAL,
             self._update_expression),
        )
        for name, enabled, interval, update in schedule:
            if enabled and now - self._last_run[name] >= interval:
                self._last_run[name] = now
                self._run_feature(name, update, small, frame, scale, events)

        return FrameResult(self._draw(small, time.monotonic()), events)

    def read_text_now(self, frame):
        """Read text in this frame immediately, interrupting current speech."""
        lines = self.reader.read(frame)
        self._set_text_lines(lines, self._scale_for(frame), ON_DEMAND_BOX_SECONDS)

        text = speakable_text(lines, max_chars=config.OCR_MAX_CHARS)
        sentence = f"Text reads: {text}" if text else "No text found."
        self.speaker.speak(sentence, interrupt=True)
        if text:
            self._remember_text(text, time.monotonic())
        return sentence

    def _run_feature(self, name, update, small, frame, scale, events):
        start = time.perf_counter()
        try:
            update(small, frame, scale, events)
        except Exception as error:
            self._disable(name)
            events.append(f"Warning: {name} was switched off after an error: {error}")
            return
        self.stats[name] = (time.perf_counter() - start) * 1000

    # ------------------------------------------------------------ features

    def _update_objects(self, small, frame, scale, events):
        height, width = small.shape[:2]
        self._detections = self.detector.detect(small)
        self._positioned = locate(self._detections, width)

        # Near warnings come first: they matter more than a list of objects.
        if self.near_enabled:
            self._hazards = find_hazards(
                self._positioned, width, height,
                close=config.PROXIMITY_CLOSE_RATIO,
                very_close=config.PROXIMITY_VERY_CLOSE_RATIO,
            )
            warning = self.hazard_warner.update(self._hazards)
            if warning:
                events.append(warning)
        else:
            self._hazards = []

        if self.objects_enabled:
            # Close obstacles are covered by the warnings, so don't announce them twice.
            handled = {(hazard.label, hazard.position) for hazard in self._hazards}
            candidates = [item for item in self._positioned
                          if (item.label, item.position) not in handled]
            message = self.announcer.announce_objects(candidates)
            if message:
                events.append(message)

    def _update_text(self, small, frame, scale, events):
        lines = self.reader.read(frame)
        self._set_text_lines(lines, scale, hold=2 * config.OCR_INTERVAL)

        text = speakable_text(lines, max_chars=config.OCR_MAX_CHARS)
        now = time.monotonic()
        if text and self._is_new_text(text, now):
            sentence = f"Text reads: {text}"
            if self.speaker.speak(sentence):
                self._remember_text(text, now)
                events.append(sentence)

    def _update_expression(self, small, frame, scale, events):
        result = self.recognizer.analyze(small)
        self._expression = result

        if result is None:
            self._face_missing += 1
            self._stable_label, self._stable_runs = None, 0
            if self._face_missing >= FACE_GONE_RUNS:
                self._announced_expression = None
            return

        self._face_missing = 0
        if result.confident:
            if result.label == self._stable_label:
                self._stable_runs += 1
            else:
                self._stable_label, self._stable_runs = result.label, 1
        else:
            self._stable_label, self._stable_runs = None, 0

        if (self._stable_runs >= config.EXPRESSION_STABLE_RUNS
                and result.label != self._announced_expression):
            sentence = describe_expression(result)
            if self.announcer.announce_message(
                    f"expression:{result.label}", sentence,
                    cooldown=config.EXPRESSION_COOLDOWN):
                self._announced_expression = result.label
                events.append(sentence)

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _scale_for(frame):
        return config.PROCESS_WIDTH / frame.shape[1]

    def _shrink(self, frame):
        height, width = frame.shape[:2]
        if width == config.PROCESS_WIDTH:
            return frame, 1.0
        scale = self._scale_for(frame)
        small = cv2.resize(frame, (config.PROCESS_WIDTH, round(height * scale)),
                           interpolation=cv2.INTER_AREA)
        return small, scale

    def _set_text_lines(self, lines, scale, hold):
        self._text_lines = [
            TextLine(line.text, line.confidence, tuple(int(v * scale) for v in line.box))
            for line in lines
        ]
        self._text_lines_until = time.monotonic() + hold

    def _is_new_text(self, text, now):
        self._spoken_texts = [(old, when) for old, when in self._spoken_texts
                              if now - when < config.OCR_COOLDOWN]
        normalised = " ".join(text.lower().split())
        return all(SequenceMatcher(None, normalised, old).ratio() < config.OCR_SIMILARITY
                   for old, _ in self._spoken_texts)

    def _remember_text(self, text, now):
        self._spoken_texts.append((" ".join(text.lower().split()), now))

    def _draw(self, small, now):
        annotated = small.copy()

        if self.objects_enabled:
            annotated = draw_detections(annotated, self._detections)
            for item in self._positioned:
                x1, _, _, y2 = item.detection.box
                cv2.putText(annotated, item.position, (x1 + 4, y2 - 8),
                            FONT, 0.55, (0, 255, 255), 2)

        if self.near_enabled:
            for hazard in self._hazards:
                color = (0, 0, 255) if hazard.level == VERY_CLOSE else (0, 140, 255)
                x1, y1, x2, y2 = hazard.box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
                cv2.putText(annotated, f"{hazard.label.upper()} {hazard.level.upper()}",
                            (x1 + 6, max(y1 + 24, 24)), FONT, 0.7, color, 2)

        if now < self._text_lines_until:
            annotated = draw_text_boxes(annotated, self._text_lines)
        return draw_expression(annotated, self._expression)


# ---------------------------------------------------------------- live test

def _state(enabled):
    return "ON" if enabled else "off"


def _run_pipeline_test():
    from src.camera import Camera

    print("Loading models...")
    pipeline = VisionPipeline()
    for warning in pipeline.warnings:
        print(f"Warning: {warning}")
    pipeline.set_enabled(objects=True, near=True)

    print("Keys: 1 objects | 2 text | 3 expression | 4 near warnings | "
          "SPACE read text now | x stop speech | q quit")

    window_name = "VisionMate - pipeline test"
    fps, last_time = 0.0, time.time()

    try:
        with Camera(index=config.CAMERA_INDEX, width=config.CAMERA_WIDTH,
                    height=config.CAMERA_HEIGHT, mirror=config.CAMERA_MIRROR) as camera:
            while True:
                frame = camera.read()
                result = pipeline.process(frame)
                for event in result.events:
                    print(event if event.startswith("Warning") else f"Spoken: {event}")

                now = time.time()
                instant = 1.0 / max(now - last_time, 1e-6)
                fps = 0.9 * fps + 0.1 * instant if fps else instant
                last_time = now

                display = result.frame
                hud = [
                    f"[1] objects {_state(pipeline.objects_enabled)}  "
                    f"[2] text {_state(pipeline.text_enabled)}  "
                    f"[3] expression {_state(pipeline.expression_enabled)}  "
                    f"[4] near {_state(pipeline.near_enabled)}",
                    f"{fps:.0f} FPS | " + " ".join(
                        f"{name} {ms:.0f}ms" for name, ms in pipeline.stats.items())
                    + (" | speaking" if pipeline.speaker.is_speaking else " | silent"),
                ]
                for row, line in enumerate(hud):
                    cv2.putText(display, line, (10, 22 + row * 22), FONT, 0.5, (0, 255, 0), 1)
                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("1"):
                    pipeline.set_enabled(objects=not pipeline.objects_enabled)
                elif key == ord("2"):
                    pipeline.set_enabled(text=not pipeline.text_enabled)
                elif key == ord("3"):
                    if pipeline.expression_available:
                        pipeline.set_enabled(expression=not pipeline.expression_enabled)
                    else:
                        print("Expression recognition is unavailable (see warning above).")
                elif key == ord("4"):
                    pipeline.set_enabled(near=not pipeline.near_enabled)
                elif key == 32:
                    try:
                        print(f"Read now: {pipeline.read_text_now(frame)}")
                    except Exception as error:
                        print(f"Text error: {error}")
                elif key == ord("x"):
                    pipeline.stop_speech()
    finally:
        pipeline.close()
        cv2.destroyAllWindows()
    print("Pipeline test finished.")


if __name__ == "__main__":
    try:
        _run_pipeline_test()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)