import time
import numpy as np

class InputManager:
    """
    Handles simulated audio and video capture for processing.
    In a real system, this would use PyAudio and OpenCV.
    """
    def __init__(self, config):
        self.camera_index = config.get('camera_index', 0)
        self.sample_rate = config.get('sample_rate', 16000)
        print(f"InputManager initialized. Camera: {self.camera_index}, Sample Rate: {self.sample_rate}")

    def capture_frame(self):
        """
        Simulates capturing a video frame (e.g., a numpy array representing the image).
        """
        # Placeholder: Return a mock frame data (e.g., 640x480 grayscale image)
        mock_frame = np.random.randint(0, 256, size=(480, 640), dtype=np.uint8)
        print("InputManager: Captured mock video frame.")
        return mock_frame

    def capture_audio_chunk(self, duration_s=2):
        """
        Simulates capturing a chunk of audio.
        """
        # Placeholder: Return a mock audio data (e.g., 2 seconds of random PCM data)
        samples = self.sample_rate * duration_s
        mock_audio = np.random.normal(0, 0.1, samples).astype(np.float32)
        print(f"InputManager: Captured mock audio chunk ({duration_s}s).")
        return mock_audio

    def close(self):
        """Clean up resources (like camera/mic streams)."""
        print("InputManager: Resources cleaned up.")
