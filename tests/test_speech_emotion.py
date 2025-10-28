import unittest
import numpy as np
from core.offline_utils import OfflineUtils
from modules.emotion_detection.speech_emotion import SpeechEmotionDetector

class MockOfflineUtils(OfflineUtils):
    def load_model(self, model_name):
        return {'simulated_model': True}

class TestSpeechEmotion(unittest.TestCase):
    def setUp(self):
        self.detector = SpeechEmotionDetector(MockOfflineUtils())

    def test_detection_output_format(self):
        """Test if the detector returns the correct dictionary structure."""
        mock_audio = np.random.rand(16000) # 1 second of audio
        result = self.detector.detect(mock_audio)
        
        self.assertIsInstance(result, dict)
        self.assertIn('emotion', result)
        self.assertIn('confidence', result)
        self.assertIsInstance(result['emotion'], str)
        self.assertIsInstance(result['confidence'], float)

    def test_empty_audio_input(self):
        """Test behavior with empty audio input."""
        result = self.detector.detect(np.array([]))
        # Expecting a fallback to 'neutral' or a default emotion with low confidence
        self.assertIn(result['emotion'], self.detector.emotion_labels + ['neutral'])

if __name__ == '__main__':
    unittest.main()
