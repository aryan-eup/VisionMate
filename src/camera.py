"""Camera capture: built-in camera, iPhone (Continuity Camera) or a phone video stream."""

import sys
import threading
import time

import cv2

STREAM_TIMEOUT_MS = 5000


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read."""


def is_stream_url(source):
    return isinstance(source, str) and "://" in source


def list_cameras(max_index=5):
    """Find camera indexes that deliver frames. Returns (index, width, height) tuples."""
    found = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if capture.isOpened():
            ok, frame = capture.read()
            if ok and frame is not None:
                found.append((index, frame.shape[1], frame.shape[0]))
        capture.release()
    return found


class _LatestFrameReader:
    """Reads a network stream on a background thread and keeps only the newest frame."""

    def __init__(self, capture):
        self._capture = capture
        self._condition = threading.Condition()
        self._frame = None
        self._counter = 0
        self._returned = 0
        self._error = None
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._stopped:
            ok, frame = self._capture.read()
            with self._condition:
                if not ok or frame is None:
                    if not self._stopped:
                        self._error = "The video stream stopped or was lost."
                    self._condition.notify_all()
                    return
                self._frame = frame
                self._counter += 1
                self._condition.notify_all()

    def read(self, timeout=5.0):
        """Wait for a frame newer than the last one returned."""
        with self._condition:
            arrived = self._condition.wait_for(
                lambda: self._counter != self._returned or self._error is not None,
                timeout,
            )
            if not arrived:
                raise CameraError("Timed out waiting for a video frame from the stream.")
            if self._counter == self._returned:
                raise CameraError(self._error or "The video stream stopped.")
            self._returned = self._counter
            return self._frame

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=STREAM_TIMEOUT_MS / 1000 + 1)


class Camera:
    def __init__(self, index=0, width=640, height=480, mirror=True):
        # `index` is a camera number (0 = built-in camera) or a stream URL
        # such as "http://192.168.1.23:8080/video".
        self.index = index
        self.width = width      # ignored for streams: the phone app sets the size
        self.height = height
        # Mirroring suits a camera facing you (laptop). Use False for a rear camera.
        self.mirror = mirror
        self._cap = None
        self._reader = None

    @property
    def is_running(self):
        return self._cap is not None

    def start(self):
        if self._cap is not None:
            return

        if is_stream_url(self.index):
            cap = self._open_stream(self.index)
            if not cap.isOpened():
                cap.release()
                raise CameraError(
                    f"Could not open the video stream {self.index}. Check that the "
                    "phone app is running, that the phone and Mac are on the same "
                    "network, and that the address is correct."
                )
            self._reader = _LatestFrameReader(cap)
            self._reader.start()
        else:
            cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap.release()
                raise CameraError(
                    f"Could not open camera {self.index}. Check camera permission "
                    "in System Settings > Privacy & Security > Camera, or try "
                    "another camera index."
                )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap

    @staticmethod
    def _open_stream(url):
        try:
            return cv2.VideoCapture(
                url, cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_TIMEOUT_MS,
                 cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_TIMEOUT_MS],
            )
        except (TypeError, cv2.error):
            return cv2.VideoCapture(url)

    def read(self):
        """Return one BGR frame as a NumPy array."""
        if self._cap is None:
            raise CameraError("Camera is not started. Call start() first.")

        if self._reader is not None:
            frame = self._reader.read()
        else:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                raise CameraError("Failed to read a frame from the camera.")

        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def stop(self):
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


def _run_camera_test(source, mirror):
    window_name = "VisionMate - camera test (press q to quit)"
    fps = 0.0
    last_time = time.time()

    with Camera(index=source, mirror=mirror) as camera:
        first_frame = camera.read()
        height, width = first_frame.shape[:2]
        print(f"Camera {source} opened: {width}x{height} (mirror {'on' if mirror else 'off'})")

        while True:
            frame = camera.read()

            now = time.time()
            instant_fps = 1.0 / max(now - last_time, 1e-6)
            fps = 0.9 * fps + 0.1 * instant_fps if fps else instant_fps
            last_time = now

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("Camera released. Test finished.")


if __name__ == "__main__":
    arguments = sys.argv[1:]

    if arguments and arguments[0] == "--list":
        cameras = list_cameras()
        if not cameras:
            print("No cameras found.")
        for index, width, height in cameras:
            print(f"Camera {index}: {width}x{height}")
        print("Test one with: python -m src.camera <index> --no-mirror")
        sys.exit(0)

    mirror = "--no-mirror" not in arguments
    positional = [a for a in arguments if not a.startswith("--")]
    source = positional[0] if positional else 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    try:
        _run_camera_test(source, mirror)
    except CameraError as error:
        print(f"Camera error: {error}")
        sys.exit(1)