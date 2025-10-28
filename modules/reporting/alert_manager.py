import time

class AlertManager:
    """
    Manages critical alerts when high stress or acute negative emotion thresholds are crossed.
    """
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.last_alert_time = 0
        self.alert_cooldown_s = 300 # 5-minute cooldown for critical alerts

    def check_critical_state(self, is_critically_stressed, fused_emotion):
        """
        Checks if a critical alert should be triggered.
        """
        current_time = time.time()
        
        # Cooldown check
        if (current_time - self.last_alert_time) < self.alert_cooldown_s:
            return {'alert_triggered': False, 'message': None}

        # 1. Critical Stress Flag
        if is_critically_stressed:
            self.last_alert_time = current_time
            return {
                'alert_triggered': True,
                'message': "CRITICAL STRESS ALERT: Sustained high stress detected. Please take immediate action."
            }
        
        # 2. Acute High Negative Emotion Check
        acute_negative = ['anger', 'fear']
        emotion = fused_emotion.get('fused_emotion')
        confidence = fused_emotion.get('score', 0.0)
        
        if emotion in acute_negative and confidence >= self.thresholds['emotion_specific'].get(emotion, 0.9):
            self.last_alert_time = current_time
            return {
                'alert_triggered': True,
                'message': f"ACUTE {emotion.upper()} DETECTED: Very high confidence in a volatile emotional state. Initiating supportive dialogue."
            }

        return {'alert_triggered': False, 'message': None}
