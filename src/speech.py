"""Text-to-speech using the native macOS `say` command."""

import shutil
import subprocess
import time


class SpeechError(RuntimeError):
    """Raised when speech output cannot be started."""


class Speaker:
    MAX_CHARS = 2000

    def __init__(self, rate=170, voice=None):
        if shutil.which("say") is None:
            raise SpeechError("The macOS 'say' command was not found.")
        self.rate = rate      # words per minute
        self.voice = voice    # None = system default voice
        self._process = None

    @property
    def is_speaking(self):
        return self._process is not None and self._process.poll() is None

    def speak(self, text, interrupt=False):
        """Start speaking. Returns True if speech started, False if skipped."""
        text = " ".join(text.split())[: self.MAX_CHARS]
        if not text:
            return False

        if self.is_speaking:
            if not interrupt:
                return False
            self.stop()

        command = ["say", "-r", str(self.rate), "-f", "-"]
        if self.voice:
            command += ["-v", self.voice]

        try:
            # Text is sent through stdin so odd characters or a leading "-"
            # can never be mistaken for command-line options.
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
            self._process.stdin.write(text)
            self._process.stdin.close()
        except OSError as error:
            self._process = None
            raise SpeechError(f"Could not start speech: {error}") from error
        return True

    def wait(self, timeout=None):
        """Block until speech finishes. Returns True if it finished."""
        if self._process is None:
            return True
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def stop(self):
        """Silence current speech immediately."""
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None


def _run_speech_test():
    speaker = Speaker()
    long_text = (
        "This is a deliberately long sentence that keeps going, so that we "
        "can test interrupting and stopping the speech output."
    )

    print("1) Normal message...")
    print("   started:", speaker.speak("VisionMate speech test. Person detected on the left."))
    speaker.wait()

    print("2) Speaking while busy (second message should be dropped)...")
    speaker.speak(long_text)
    time.sleep(0.5)
    print("   second message started:", speaker.speak("This should never be heard."))
    speaker.wait()

    print("3) Interrupt...")
    speaker.speak(long_text)
    time.sleep(1.5)
    started = speaker.speak("Interrupted. Chair detected in the centre.", interrupt=True)
    print("   interrupting message started:", started)
    speaker.wait()

    print("4) Stop...")
    speaker.speak(long_text)
    time.sleep(1.5)
    speaker.stop()
    print("   is_speaking after stop:", speaker.is_speaking)

    print("Speech test finished.")


if __name__ == "__main__":
    try:
        _run_speech_test()
    except SpeechError as error:
        print(f"Speech error: {error}")
        raise SystemExit(1)