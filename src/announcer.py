"""Decides what gets spoken and when: filtering, de-duplication, cooldowns."""

import sys
import time


class Announcer:
    def __init__(self, speaker, object_cooldown=8.0, min_consecutive=2,
                 max_objects_per_message=3, clock=time.monotonic):
        self.speaker = speaker
        self.object_cooldown = object_cooldown
        self.min_consecutive = min_consecutive
        self.max_objects_per_message = max_objects_per_message
        self._clock = clock
        self._streaks = {}       # (label, position) -> consecutive detection runs
        self._last_spoken = {}   # key -> time it was last spoken

    def reset(self):
        """Forget everything, e.g. when a feature is switched off and on."""
        self._streaks.clear()
        self._last_spoken.clear()

    def announce_objects(self, positioned):
        """Call once per detection run with PositionedDetection items.

        Returns the sentence that started speaking, or None.
        """
        unique = {}
        for item in positioned:
            unique.setdefault((item.label, item.position), item)

        # Objects that were not seen this run lose their streak.
        self._streaks = {key: self._streaks.get(key, 0) + 1 for key in unique}

        now = self._clock()
        ready = [
            key for key in unique
            if self._streaks[key] >= self.min_consecutive
            and self._is_due(("object", key), now, self.object_cooldown)
        ]
        ready = ready[: self.max_objects_per_message]
        if not ready:
            return None

        message = " ".join(unique[key].sentence for key in ready)
        if not self.speaker.speak(message):
            return None  # speaker busy: nothing is marked, so we retry next run

        for key in ready:
            self._last_spoken[("object", key)] = now
        return message

    def announce_message(self, key, text, cooldown=20.0, interrupt=False):
        """Speak a one-off message (OCR text, expression) unless repeated too soon."""
        now = self._clock()
        if not self._is_due(("message", key), now, cooldown):
            return False
        if not self.speaker.speak(text, interrupt=interrupt):
            return False
        self._last_spoken[("message", key)] = now
        return True

    def _is_due(self, key, now, cooldown):
        last = self._last_spoken.get(key)
        return last is None or now - last >= cooldown


# ---------------------------------------------------------------- tests

class _FakeSpeaker:
    def __init__(self):
        self.spoken = []
        self.busy = False

    def speak(self, text, interrupt=False):
        if self.busy and not interrupt:
            return False
        self.spoken.append(text)
        return True


class _FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _run_unit_checks():
    from types import SimpleNamespace

    from src.position import locate

    left_x, centre_x, right_x = 100, 320, 540  # in a 640-pixel-wide frame

    def detections(*items):
        raw = [SimpleNamespace(label=label, box=(x - 10, 0, x + 10, 50))
               for label, x in items]
        return locate(raw, 640)

    def make():
        speaker, clock = _FakeSpeaker(), _FakeClock()
        announcer = Announcer(speaker, object_cooldown=8.0, min_consecutive=2, clock=clock)
        return announcer, speaker, clock

    person_left = detections(("person", left_x))
    person_centre = detections(("person", centre_x))

    # 1. Needs two runs in a row before speaking.
    announcer, speaker, clock = make()
    assert announcer.announce_objects(person_left) is None
    assert announcer.announce_objects(person_left) == "Person detected on the left."

    # 2. Not repeated during the cooldown, spoken again after it.
    clock.t += 3
    assert announcer.announce_objects(person_left) is None
    clock.t += 6
    assert announcer.announce_objects(person_left) == "Person detected on the left."

    # 3. Moving to another zone is announced as new information.
    assert announcer.announce_objects(person_centre) is None
    assert announcer.announce_objects(person_centre) == "Person detected in the centre."

    # 4. A one-run flicker never gets announced.
    announcer, speaker, clock = make()
    assert announcer.announce_objects(person_left) is None
    assert announcer.announce_objects([]) is None
    assert announcer.announce_objects(person_left) is None
    assert announcer.announce_objects(person_left) == "Person detected on the left."

    # 5. A busy speaker does not cause a message to be lost.
    announcer, speaker, clock = make()
    speaker.busy = True
    announcer.announce_objects(person_left)
    assert announcer.announce_objects(person_left) is None
    assert speaker.spoken == []
    speaker.busy = False
    assert announcer.announce_objects(person_left) == "Person detected on the left."

    # 6. At most three objects per message, the rest follow.
    announcer, speaker, clock = make()
    many = detections(("person", left_x), ("chair", centre_x),
                      ("bottle", right_x), ("dog", right_x - 50))
    announcer.announce_objects(many)
    assert announcer.announce_objects(many) == (
        "Person detected on the left. Chair detected in the centre. "
        "Bottle detected on the right."
    )
    assert announcer.announce_objects(many) == "Dog detected on the right."
    assert announcer.announce_objects(many) is None

    # 7. Two people in the same zone are announced once.
    announcer, speaker, clock = make()
    two_people = detections(("person", left_x), ("person", left_x + 20))
    announcer.announce_objects(two_people)
    assert announcer.announce_objects(two_people) == "Person detected on the left."

    # 8. One-off messages have their own cooldown per key.
    announcer, speaker, clock = make()
    assert announcer.announce_message("text:hello", "Text reads hello.", cooldown=20) is True
    assert announcer.announce_message("text:hello", "Text reads hello.", cooldown=20) is False
    assert announcer.announce_message("expression:happy", "Face looks happy.", cooldown=20) is True
    clock.t += 21
    assert announcer.announce_message("text:hello", "Text reads hello.", cooldown=20) is True

    # 9. Reset forgets everything.
    announcer.reset()
    assert announcer.announce_message("text:hello", "Text reads hello.", cooldown=20) is True

    print("Unit checks passed.")


def _run_live_test():
    import cv2

    from src.camera import Camera
    from src.object_detector import ObjectDetector, draw_detections
    from src.position import locate
    from src.speech import Speaker

    detection_interval = 0.4  # seconds between YOLO runs

    print("Loading model...")
    detector = ObjectDetector()
    speaker = Speaker()
    announcer = Announcer(speaker)
    window_name = "VisionMate - announcer test (press q to quit)"

    detections = []
    last_detection = 0.0

    try:
        with Camera() as camera:
            while True:
                frame = camera.read()

                now = time.monotonic()
                if now - last_detection >= detection_interval:
                    last_detection = now
                    detections = detector.detect(frame)
                    spoken = announcer.announce_objects(locate(detections, frame.shape[1]))
                    if spoken:
                        print(f"Spoken: {spoken}")

                annotated = draw_detections(frame, detections)
                status = "speaking" if speaker.is_speaking else "silent"
                cv2.putText(annotated, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(window_name, annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        speaker.stop()
        cv2.destroyAllWindows()
    print("Announcer test finished.")


if __name__ == "__main__":
    try:
        _run_unit_checks()
        _run_live_test()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)