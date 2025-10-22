# MAITRI System Constants

# Standardized Emotion Labels for MAITRI System (Fix 8)
# Used across facial, speech, and fusion modules for consistency.
EMOTION_LABELS = ['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise', 'disgust']

# Feature Extraction Constants (Fix 9 related)
# Must match the expected input size of the respective AI models.
VISUAL_FEATURE_SIZE = 128
AUDIO_FEATURE_SIZE = 80 # (e.g., 20 MFCCs * 4 frames)
