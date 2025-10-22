import time
import numpy as np

class FatigueDetector:
    """
    Analyzes visual and auditory cues (e.g., blink rate, head pose, speaking speed)
    to estimate user fatigue level.
    """
    def __init__(self, thresholds):
        self.history = [] # To track recent fatigue scores
        self.max_history = 30 # Track last 30 measurements
        self.fatigue_mean_threshold = thresholds.get('fatigue_mean_threshold', 0.65)

    def check_fatigue_cues(self, frame_data, audio_data):
        """
        Simulates analysis of visual and audio cues.
        """
        # --- Visual Cues Simulation (e.g., low head tilt, slow movement)
        visual_fatigue_score = np.random.uniform(0.0, 1.0) * 0.2 # low contribution

        # --- Audio Cues Simulation (e.g., slow speech rate, low pitch variability)
        audio_fatigue_score = np.random.uniform(0.0, 1.0) * 0.3 # moderate contribution

        # --- Combined behavioral cues (e.g., time since last break)
        # For simplicity, let's make it time-dependent
        current_hour = time.localtime().tm_hour
        time_based_score = 0.0
        if current_hour >= 20 or current_hour <= 7: # Evening/Night/Early Morning
            time_based_score = np.random.uniform(0.4, 0.8) # Higher base score

        total_fatigue_score = visual_fatigue_score + audio_fatigue_score + time_based_score

        # Normalize to 0-1 range (simplified for simulation)
        total_fatigue_score = min(total_fatigue_score, 1.0)

        self.history.append(total_fatigue_score)
        self.history = self.history[-self.max_history:]

        is_fatigued = np.mean(self.history) > self.fatigue_mean_threshold

        return {'score': round(total_fatigue_score, 4), 'is_fatigued': is_fatigued}

