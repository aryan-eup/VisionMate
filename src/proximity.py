"""Near-obstacle warnings from bounding-box size (an approximation, not true depth)."""

import sys
import time
from dataclasses import dataclass

from src.position import CENTRE, LEFT, RIGHT, locate

# Objects that can block a walking path (COCO class names).
HAZARD_LABELS = frozenset({
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "bench", "chair",
    "couch", "dining table", "potted plant", "dog", "fire hydrant", "parking meter",
})

CLOSE_RATIO = 0.40        # object fills at least this share of the view -> "close"
VERY_CLOSE_RATIO = 0.65   # ... and this share -> "very close"

CLOSE, VERY_CLOSE = "close", "very close"
_RANK = {CLOSE: 1, VERY_CLOSE: 2}
_WHERE = {LEFT: "on the left", CENTRE: "in the centre", RIGHT: "on the right"}


@dataclass(frozen=True)
class Hazard:
    label: str
    position: str   # left / centre / right
    level: str      # "close" or "very close"
    ratio: float    # how much of the view the object fills (0-1)
    box: tuple

    @property
    def rank(self):
        return _RANK[self.level]

    @property
    def sentence(self):
        text = f"{self.label.capitalize()} {self.level} {_WHERE[self.position]}."
        return f"Warning! {text}" if self.level == VERY_CLOSE else text


def fill_ratio(box, frame_width, frame_height):
    """Share of the view an object fills: the larger of its height and width share."""
    x1, y1, x2, y2 = box
    return max((y2 - y1) / frame_height, (x2 - x1) / frame_width)


def classify(ratio, close=CLOSE_RATIO, very_close=VERY_CLOSE_RATIO):
    if ratio >= very_close:
        return VERY_CLOSE
    if ratio >= close:
        return CLOSE
    return None


def find_hazards(positioned, frame_width, frame_height,
                 close=CLOSE_RATIO, very_close=VERY_CLOSE_RATIO):
    """Close obstacles among PositionedDetection items, most urgent first."""
    hazards = []
    for item in positioned:
        if item.label not in HAZARD_LABELS:
            continue
        ratio = fill_ratio(item.detection.box, frame_width, frame_height)
        level = classify(ratio, close, very_close)
        if level:
            hazards.append(Hazard(item.label, item.position, level, ratio,
                                  item.detection.box))
    hazards.sort(key=lambda hazard: (hazard.rank, hazard.ratio), reverse=True)
    return hazards


class HazardWarner:
    """Decides when a warning is spoken: streaks, cooldowns and escalation."""

    def __init__(self, speaker, cooldown=4.0, min_consecutive=2, clock=time.monotonic):
        self.speaker = speaker
        self.cooldown = cooldown
        self.min_consecutive = min_consecutive
        self._clock = clock
        self._streaks = {}   # (label, position) -> consecutive runs seen
        self._spoken = {}    # (label, position) -> (rank, time last warned)

    def reset(self):
        self._streaks.clear()
        self._spoken.clear()

    def update(self, hazards):
        """Call once per detection run with hazards sorted most urgent first.

        Returns the sentence that started speaking, or None.
        """
        current = {}
        for hazard in hazards:
            current.setdefault((hazard.label, hazard.position), hazard)

        # Objects not seen this run lose their streak.
        self._streaks = {key: self._streaks.get(key, 0) + 1 for key in current}

        now = self._clock()
        ready = [
            hazard for key, hazard in current.items()
            if self._streaks[key] >= self.min_consecutive
            and self._is_due(key, hazard.rank, now)
        ]
        if not ready:
            return None

        hazard = ready[0]  # the most urgent one
        if not self.speaker.speak(hazard.sentence, interrupt=hazard.level == VERY_CLOSE):
            return None    # speaker busy: nothing recorded, so we retry next run

        self._spoken[(hazard.label, hazard.position)] = (hazard.rank, now)
        return hazard.sentence

    def _is_due(self, key, rank, now):
        record = self._spoken.get(key)
        if record is None:
            return True
        last_rank, last_time = record
        return rank > last_rank or now - last_time >= self.cooldown


# ---------------------------------------------------------------- tests

class _FakeSpeaker:
    def __init__(self):
        self.calls = []   # (text, interrupt)
        self.busy = False

    def speak(self, text, interrupt=False):
        if self.busy and not interrupt:
            return False
        self.calls.append((text, interrupt))
        return True


class _FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _run_unit_checks():
    from types import SimpleNamespace

    left, centre, right = 100, 320, 540  # x positions in a 640-pixel-wide frame

    def hazards(*items):
        """items are (label, x_centre, fill_ratio) in a 640x360 frame."""
        raw = [SimpleNamespace(label=label,
                               box=(x - 10, 360 - round(fill * 360), x + 10, 360))
               for label, x, fill in items]
        return find_hazards(locate(raw, 640), 640, 360)

    def make():
        speaker, clock = _FakeSpeaker(), _FakeClock()
        return HazardWarner(speaker, cooldown=4.0, min_consecutive=2, clock=clock), speaker, clock

    # 1. Ratio and classification boundaries.
    assert fill_ratio((0, 0, 320, 90), 640, 360) == 0.5
    assert fill_ratio((0, 0, 100, 360), 640, 360) == 1.0
    assert classify(0.39) is None
    assert classify(0.40) == CLOSE
    assert classify(0.64) == CLOSE
    assert classify(0.65) == VERY_CLOSE

    # 2. Only close obstacles count, most urgent first.
    assert hazards(("bottle", centre, 0.9)) == []
    assert hazards(("person", centre, 0.3)) == []
    ranked = hazards(("chair", left, 0.5), ("car", right, 0.8))
    assert [h.label for h in ranked] == ["car", "chair"]

    # 3. Sentences.
    assert hazards(("person", centre, 0.5))[0].sentence == "Person close in the centre."
    assert hazards(("dog", left, 0.8))[0].sentence == "Warning! Dog very close on the left."
    assert hazards(("bus", right, 0.5))[0].sentence == "Bus close on the right."

    close_person = hazards(("person", centre, 0.5))
    very_close_person = hazards(("person", centre, 0.8))

    # 4. Needs two runs in a row; cooldown; repeats after it.
    warner, speaker, clock = make()
    assert warner.update(close_person) is None
    assert warner.update(close_person) == "Person close in the centre."
    clock.t += 2
    assert warner.update(close_person) is None
    clock.t += 3
    assert warner.update(close_person) == "Person close in the centre."

    # 5. Escalation is immediate and interrupts; de-escalation is quiet.
    warner, speaker, clock = make()
    warner.update(close_person)
    warner.update(close_person)
    clock.t += 1
    assert warner.update(very_close_person) == "Warning! Person very close in the centre."
    assert speaker.calls[-1][1] is True, "very close should interrupt"
    assert speaker.calls[0][1] is False, "close should not interrupt"
    clock.t += 1
    assert warner.update(close_person) is None
    clock.t += 5
    assert warner.update(close_person) == "Person close in the centre."

    # 6. A one-run flicker is ignored.
    warner, speaker, clock = make()
    assert warner.update(close_person) is None
    assert warner.update([]) is None
    assert warner.update(close_person) is None
    assert warner.update(close_person) == "Person close in the centre."

    # 7. A busy speaker does not lose the warning.
    warner, speaker, clock = make()
    speaker.busy = True
    warner.update(close_person)
    assert warner.update(close_person) is None
    assert speaker.calls == []
    speaker.busy = False
    assert warner.update(close_person) == "Person close in the centre."

    # 8. Several hazards: the most urgent is spoken first, the rest follow.
    warner, speaker, clock = make()
    both = hazards(("chair", left, 0.5), ("car", right, 0.8))
    warner.update(both)
    assert warner.update(both) == "Warning! Car very close on the right."
    assert warner.update(both) == "Chair close on the left."
    assert warner.update(both) is None

    # 9. Reset forgets everything.
    warner.reset()
    assert warner.update(both) is None
    assert warner.update(both) == "Warning! Car very close on the right."

    print("Unit checks passed.")


def _run_live_test():
    import cv2

    from src import config
    from src.camera import Camera
    from src.object_detector import ObjectDetector
    from src.speech import Speaker

    print("Loading model...")
    detector = ObjectDetector()
    speaker = Speaker()
    warner = HazardWarner(speaker)
    window_name = "VisionMate - proximity test (press q to quit)"
    interval, last_run, last_print = 0.3, 0.0, 0.0
    detections, width, height = [], config.PROCESS_WIDTH, 360

    try:
        with Camera(index=config.CAMERA_INDEX, width=config.CAMERA_WIDTH,
                    height=config.CAMERA_HEIGHT, mirror=config.CAMERA_MIRROR) as camera:
            while True:
                frame = camera.read()
                scale = config.PROCESS_WIDTH / frame.shape[1]
                small = cv2.resize(frame, (config.PROCESS_WIDTH, round(frame.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)
                height, width = small.shape[:2]

                now = time.monotonic()
                if now - last_run >= interval:
                    last_run = now
                    detections = detector.detect(small)
                    hazards = find_hazards(locate(detections, width), width, height)
                    spoken = warner.update(hazards)
                    if spoken:
                        print(f"Spoken: {spoken}")
                    if now - last_print >= 1.0:
                        last_print = now
                        print(" | ".join(f"{h.label} {h.level} {h.position} ({h.ratio:.2f})"
                                         for h in hazards) or "no hazards")

                display = small.copy()
                for detection in detections:
                    ratio = fill_ratio(detection.box, width, height)
                    level = classify(ratio) if detection.label in HAZARD_LABELS else None
                    color = ((0, 0, 255) if level == VERY_CLOSE
                             else (0, 200, 255) if level == CLOSE else (160, 160, 160))
                    x1, y1, x2, y2 = detection.box
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    label = f"{detection.label} {ratio:.2f}" + (f" {level.upper()}" if level else "")
                    cv2.putText(display, label, (x1 + 3, max(y1 + 16, 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.putText(display, f"close >= {CLOSE_RATIO:.2f} | very close >= {VERY_CLOSE_RATIO:.2f}",
                            (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        speaker.stop()
        cv2.destroyAllWindows()
    print("Proximity test finished.")


if __name__ == "__main__":
    try:
        _run_unit_checks()
        _run_live_test()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)