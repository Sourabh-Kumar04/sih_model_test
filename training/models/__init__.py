"""
MAITRI Model Package
"""

from .facial_emotion_model import (
    FacialEmotionCNN,
    FacialEmotionResNet,
    FacialEmotionMobileNet,
    get_model as get_facial_model
)

from .speech_emotion_model import (
    SpeechEmotionLSTM,
    SpeechEmotionCNN,
    SpeechEmotionCNNLSTM,
    SpeechEmotionAttention,
    get_model as get_speech_model
)

from .multimodal_fusion_model import (
    LateFusionModel,
    EarlyFusionModel,
    AttentionFusionModel,
    ConfidenceWeightedFusion,
    MultimodalFusionModel,
    get_fusion_model
)

__all__ = [
    'FacialEmotionCNN',
    'FacialEmotionResNet',
    'FacialEmotionMobileNet',
    'get_facial_model',
    'SpeechEmotionLSTM',
    'SpeechEmotionCNN',
    'SpeechEmotionCNNLSTM',
    'SpeechEmotionAttention',
    'get_speech_model',
    'LateFusionModel',
    'EarlyFusionModel',
    'AttentionFusionModel',
    'ConfidenceWeightedFusion',
    'MultimodalFusionModel',
    'get_fusion_model'
]
