import unittest
import os
import json
from modules.conversation.dialogue_manager import DialogueManager

# Define a temporary response library path for testing
TEST_JSON_PATH = 'test_response_library.json'

class TestDialogueManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a mock response library file for setup
        mock_responses = {
          "happy": ["That's great!"],
          "sad": ["I'm here for you."],
          "test_emotion": ["Response A", "Response B"]
        }
        with open(TEST_JSON_PATH, 'w') as f:
            json.dump(mock_responses, f)

    @classmethod
    def tearDownClass(cls):
        # Clean up the mock file
        if os.path.exists(TEST_JSON_PATH):
            os.remove(TEST_JSON_PATH)

    def setUp(self):
        self.manager = DialogueManager(TEST_JSON_PATH)

    def test_response_loading(self):
        """Test if the responses were loaded correctly."""
        self.assertIn('sad', self.manager.response_library)
        self.assertEqual(len(self.manager.response_library['happy']), 1)

    def test_response_generation_by_emotion(self):
        """Test generating a response for a known emotion."""
        response = self.manager.generate_response('HAPPY')
        self.assertEqual(response, "That's great!")

    def test_response_repetition_prevention(self):
        """Test that responses cycle if multiple options are available."""
        # This test requires an emotion with multiple responses
        r1 = self.manager.generate_response('TEST_EMOTION')
        r2 = self.manager.generate_response('TEST_EMOTION')
        r3 = self.manager.generate_response('TEST_EMOTION')
        
        # With only A and B, r3 must be r1
        self.assertIn(r1, ["Response A", "Response B"])
        self.assertNotEqual(r1, r2)
        self.assertEqual(r3, r1)

    def test_default_response_for_unknown_emotion(self):
        """Test fallback to default/neutral response."""
        response = self.manager.generate_response('CONFUSED')
        self.assertIn(response, self.manager.response_library.get('neutral', []))

if __name__ == '__main__':
    unittest.main()
