import random

class Recommendations:
    """
    Generates practical, immediate health tips based on detected state.
    """
    def __init__(self):
        self.tip_library = {
            'stress': [
                "Try the 4-7-8 breathing technique: inhale for 4, hold for 7, exhale for 8.",
                "Step away from your screen for five minutes and stretch your neck and shoulders.",
                "Listen to one of your favorite, calming songs right now.",
                "Write down the single most urgent task. Ignore the rest for 10 minutes."
            ],
            'fatigue': [
                "Remember to hydrate! A glass of water can boost your energy.",
                "If possible, step outside and get some natural light for two minutes.",
                "Do a quick eye-stretching exercise (look left, right, up, down).",
                "Try a power nap later, but for now, stretch and move around."
            ],
            'neutral': [
                "Keep up the great work. Remember to pause and appreciate your progress.",
                "How long has it been since you last stood up? Movement helps cognitive function.",
                "Check your posture—a relaxed back helps concentration."
            ]
        }
        self.last_tip = ""

    def get_tip(self, state):
        """
        Returns a single, relevant, non-repeating recommendation.
        
        Args:
            state (str): The current detected state ('stress', 'fatigue', or an emotion).
            
        Returns:
            str: A health recommendation tip.
        """ 
        key = state.lower()
        # Check for generic health state mapping: treat specific negative emotions as 'stress'
        if key == 'sad' or key == 'angry' or key == 'fear':
            key = 'stress'
        else:
            key = 'neutral'

        tip_list = self.tip_library[key]
        new_tip = random.choice(tip_list)

        # Avoid immediate repetition
        while new_tip == self.last_tip and len(tip_list) > 1:
            new_tip = random.choice(tip_list)

        self.last_tip = new_tip
        return new_tip
