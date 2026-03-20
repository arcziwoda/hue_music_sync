"""Audio capture module — PyAudio wrapper with threaded capture."""

import logging
import threading
from collections import deque
from typing import Optional

import numpy as np
import pyaudio

from ..core.exceptions import AudioCaptureError

logger = logging.getLogger(__name__)

# PyAudio format: 16-bit signed int
FORMAT = pyaudio.paInt16
CHANNELS = 1
DTYPE = np.int16
MAX_INT16 = 32768.0  # For normalization to [-1.0, 1.0]


class AudioCapture:
    """
    Threaded audio capture from microphone or system audio.

    Captures audio in a background thread and stores frames in a ring buffer.
    Consumers call get_frame() to retrieve the latest audio data.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        buffer_size: int = 1024,
        device_index: Optional[int] = None,
        max_queue_size: int = 64,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.device_index = device_index

        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Ring buffer of normalized float32 frames
        self._frames: deque[np.ndarray] = deque(maxlen=max_queue_size)
        self._lock = threading.Lock()
        self._frame_event = threading.Event()

    def start(self) -> None:
        """Start audio capture in a background thread."""
        if self._running:
            return

        self._pa = pyaudio.PyAudio()

        device_info = self._get_device_info()
        logger.info(
            f"Opening audio: {device_info.get('name')} "
            f"@ {self.sample_rate}Hz, buffer={self.buffer_size}"
        )

        try:
            self._stream = self._pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.buffer_size,
            )
        except Exception as e:
            self._cleanup_pa()
            raise AudioCaptureError(f"Failed to open audio stream: {e}") from e

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Audio capture started")

    def stop(self) -> None:
        """Stop audio capture and release resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._cleanup_pa()
        logger.info("Audio capture stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent audio frame (non-blocking).

        Returns:
            Normalized float32 array of shape (buffer_size,) in range [-1, 1],
            or None if no frame available.
        """
        with self._lock:
            if self._frames:
                return self._frames[-1]
        return None

    def get_all_frames(self) -> list[np.ndarray]:
        """Get all buffered frames and clear the buffer."""
        with self._lock:
            frames = list(self._frames)
            self._frames.clear()
        return frames

    def wait_for_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Block until a new frame is available."""
        self._frame_event.clear()
        if self._frame_event.wait(timeout):
            return self.get_frame()
        return None

    @property
    def is_running(self) -> bool:
        return self._running

    def list_devices(self) -> list[dict]:
        """List available audio input devices."""
        pa = self._pa or pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                    "sample_rate": int(info["defaultSampleRate"]),
                })
        if not self._pa:
            pa.terminate()
        return devices

    def _capture_loop(self) -> None:
        """Background thread: read audio frames continuously."""
        while self._running:
            try:
                raw = self._stream.read(self.buffer_size, exception_on_overflow=False)
                samples = np.frombuffer(raw, dtype=DTYPE).astype(np.float32) / MAX_INT16

                with self._lock:
                    self._frames.append(samples)
                self._frame_event.set()

            except Exception as e:
                if self._running:
                    logger.error(f"Audio capture error: {e}")

    def _get_device_info(self) -> dict:
        """Get info for the selected or default device."""
        if self.device_index is not None:
            return self._pa.get_device_info_by_index(self.device_index)
        return self._pa.get_default_input_device_info()

    def _cleanup_pa(self) -> None:
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
