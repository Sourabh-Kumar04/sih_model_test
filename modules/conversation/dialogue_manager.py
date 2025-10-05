import random
import json
import os

class DialogueManager:
    """
    Manages conversational flow and selects adaptive, supportive responses
    based on the user's detected emotional state.
    """
    def __init__(self, response_library_path):
        self.response_library = self._load_responses(response_library_path)
        self.last_response = ""

    def _load_responses(self, path):
        """Loads the predefined supportive responses from a JSON file."""
        if not os.path.exists(path):
            print(f"ERROR: Response library not found at {path}. Using minimal default.")
            return {"neutral": ["Hello. How can I assist you today?"]}
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR loading response library: {e}")
            return {"neutral": ["Hello. How can I assist you today?"]}

    def generate_response(self, fused_emotion, is_stressed=False):
        """
        Generates a contextual response based on the primary emotion and stress flag.
        """
        emotion_key = fused_emotion.lower()

        if is_stressed and 'stress' in self.response_library:
            response_list = self.response_library['stress']
        elif emotion_key in self.response_library:
            response_list = self.response_library[emotion_key]
        else:
            response_list = self.response_library.get('neutral', ["I'm not sure what to say right now, but I'm here for you."])

        # Ensure we don't repeat the exact same response immediately
        new_response = random.choice(response_list)
        while new_response == self.last_response and len(response_list) > 1:
            new_response = random.choice(response_list)

        self.last_response = new_response
        return new_response
