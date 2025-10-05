import numpy as np
# from modules.emotion_detection.facial_emotion import FacialEmotionDetector
# from modules.emotion_detection.speech_emotion import SpeechEmotionDetector

class MultimodalFusionEngine:
    """
    Combines emotion predictions from facial and speech analysis
    using a weighted or rule-based fusion strategy.
    """
    def __init__(self, thresholds):
        self.thresholds = thresholds
        # Weights for fusion (can be learned or hardcoded)

        self.thresholds = thresholds
        self.high_conf = thresholds.get('fusion_high_confidence', 0.8)
        self.weights = thresholds.get('fusion_weights', {'facial': 0.6, 'speech': 0.4})
        self.supported_emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear']

    def fuse(self, facial_result, speech_result):
        """
        Performs fusion of the two results.

        Args:
            facial_result (dict): {'emotion': str, 'confidence': float}
            speech_result (dict): {'emotion': str, 'confidence': float}

        Returns:
            dict: Fused result {'fused_emotion': str, 'score': float}
        """
        f_emotion = facial_result.get('emotion')
        f_conf = facial_result.get('confidence', 0.0)
        s_emotion = speech_result.get('emotion')
        s_conf = speech_result.get('confidence', 0.0)

        print(f"Fusion Input - Facial: {f_emotion} ({f_conf:.2f}), Speech: {s_emotion} ({s_conf:.2f})")

        # Rule 1: If both agree and confidence is high, use that emotion
        if f_emotion == s_emotion and f_conf > self.high_conf and s_conf > self.high_conf:
            fused_emotion = f_emotion
            score = (f_conf + s_conf) / 2
        # Rule 2: Prioritize visual if speech is neutral and facial is not
        elif s_emotion == 'neutral' and f_emotion != 'neutral' and f_conf > self.thresholds['min_confidence']:
            fused_emotion = f_emotion
            score = f_conf * self.weights['facial']
        # Rule 3: Weighted average confidence for matching emotions
        elif f_emotion == s_emotion:
            fused_emotion = f_emotion
            score = (f_conf * self.weights['facial']) + (s_conf * self.weights['speech'])
        # Rule 4: If negative emotion is detected in *either* modality with high confidence, flag it.
        elif f_conf > self.high_conf and f_emotion in ['sad', 'angry', 'fear']:
            fused_emotion = f_emotion
            score = f_conf
        elif s_conf > self.high_conf and s_emotion in ['sad', 'angry', 'fear']:
            fused_emotion = s_emotion
            score = s_conf
        # Default: Use the one with the highest weighted score
        else:
            facial_score = f_conf * self.weights['facial']
            speech_score = s_conf * self.weights['speech']

            if facial_score >= speech_score:
                fused_emotion = f_emotion
                score = facial_score
            else:
                fused_emotion = s_emotion
                score = speech_score

        return {'fused_emotion': fused_emotion, 'score': round(score, 4)}

# Example usage (commented out)
# if __name__ == '__main__':