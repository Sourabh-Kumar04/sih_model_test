import unittest
import numpy as np
# Assuming the necessary imports can be resolved relative to the MAITRI_AI_Assistant root
# Note: Mocking is essential here as we cannot load real models or capture real frames.
from core.offline_utils import OfflineUtils
from modules.emotion_detection.facial_emotion import FacialEmotionDetector

class MockOfflineUtils(OfflineUtils):
    """Mock utility to prevent real model loading."""
    def load_model(self, model_name):
        # Return a simple object indicating a 'loaded' model for the test logic
        return {'simulated_model': True}

class TestFacialEmotion(unittest.TestCase):
    def setUp(self):
        # Initialize detector with mock loader
        self.detector = FacialEmotionDetector(MockOfflineUtils())

    def test_detection_output_format(self):
        """Test if the detector returns the correct dictionary structure."""
        mock_frame = np.random.rand(100, 100)
        result = self.detector.detect(mock_frame)
        
        self.assertIsInstance(result, dict)
        self.assertIn('emotion', result)
        self.assertIn('confidence', result)
        self.assertIsInstance(result['emotion'], str)
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

    def test_null_input_handling(self):
        """Test behavior with a null frame input."""
        result = self.detector.detect(None)
        self.assertIn(result['emotion'], self.detector.emotion_labels + ['neutral'])

if __name__ == '__main__':
    unittest.main()
