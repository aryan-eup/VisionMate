"""Webcam capture wrapper used by the whole project."""

import sys
import time

import cv2


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read."""


class Camera:
    def __init__(self, index=0, width=640, height=480, mirror=True):
        self.index = index
        self.width = width
        self.height = height
        # Mirroring makes the preview behave like a mirror, so "left" in the
        # image matches the left side of the person sitting at the laptop.
        self.mirror = mirror
        self._cap = None

    @property
    def is_running(self):
        return self._cap is not None

    def start(self):
        if self._cap is not None:
            return

        backend = getattr(cv2, "CAP_AVFOUNDATION", None)
        if backend is None:
            backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)

        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
            if not cap.isOpened():
                raise CameraError(
                    f"Could not open camera {self.index}. Check camera permission "
                    "in System Settings > Privacy & Security > Camera, or try "
                    "another camera index."
                )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap

    def read(self):
        """Return one BGR frame as a NumPy array."""
        if self._cap is None:
            raise CameraError("Camera is not started. Call start() first.")

        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("Failed to read a frame from the camera.")

        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def stop(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


def _run_camera_test(index):
    window_name = "VisionMate - camera test (press q to quit)"
    fps = 0.0
    last_time = time.time()

    with Camera(index=index) as camera:
        first_frame = camera.read()
        height, width = first_frame.shape[:2]
        print(f"Camera {index} opened: {width}x{height}")

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
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    try:
        _run_camera_test(camera_index)
    except CameraError as error:
        print(f"Camera error: {error}")
        sys.exit(1)