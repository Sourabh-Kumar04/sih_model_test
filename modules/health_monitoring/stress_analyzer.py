import time
import numpy as np

class StressAnalyzer:
    """
    Calculates a cumulative stress score based on fused negative emotions
    and physiological indicators (simulated heart rate, skin conductance).
    """
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.stress_history = []
        self.max_history = 60 # Track stress over the last 60 seconds
        self.critical_stress_threshold = thresholds.get('stress_level_high', 0.75)

    def analyze(self, fused_emotion, fatigue_score):
        """
        Calculates the current stress index.
        """
        emotion = fused_emotion.get('fused_emotion')
        score = fused_emotion.get('score', 0.0)

        stress_contribution = 0.0
        negative_emotions = ['sad', 'angry', 'fear', 'distress']

        # 1. Emotion Contribution
        if emotion in negative_emotions:
            # Scale the contribution based on confidence and specific emotion thresholds
            emotion_threshold = self.thresholds['emotion_specific'].get(emotion, self.thresholds['min_confidence'])
            if score >= emotion_threshold:
                stress_contribution += (score * 0.4) # 40% weight

        # 2. Fatigue Contribution (High fatigue increases stress sensitivity)
        stress_contribution += (fatigue_score * 0.3) # 30% weight

        # 3. Simulated Physiological Indicators (e.g., high and variable heart rate)
        simulated_hr_variance = np.random.uniform(0.0, 0.3)
        stress_contribution += (simulated_hr_variance * 0.3) # 30% weight

        current_stress_index = min(stress_contribution, 1.0)

        self.stress_history.append(current_stress_index)
        self.stress_history = self.stress_history[-self.max_history:]

        # Determine if critically stressed (average score over the last minute is high)
        average_stress = np.mean(self.stress_history)
        is_critically_stressed = average_stress > self.critical_stress_threshold

        return {
            'current_index': round(current_stress_index, 4), 
            'average_stress_60s': round(average_stress, 4),
            'is_critically_stressed': is_critically_stressed
        }
        
