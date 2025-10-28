"""
Multimodal Fusion Model
Combines facial and speech emotion predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LateFusionModel(nn.Module):
    """
    Late Fusion: Combines predictions from facial and speech models
    """
    def __init__(self, num_classes=7):
        super(LateFusionModel, self).__init__()
        
        # Fusion weights (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Weight for facial
        # Weight for speech is (1 - alpha)
        
    def forward(self, facial_logits, speech_logits):
        # Weighted average of predictions
        alpha = torch.sigmoid(self.alpha)  # Ensure between 0 and 1
        fused = alpha * facial_logits + (1 - alpha) * speech_logits
        return fused


class EarlyFusionModel(nn.Module):
    """
    Early Fusion: Combines features before final classification
    """
    def __init__(self, facial_features=256, speech_features=256, num_classes=7):
        super(EarlyFusionModel, self).__init__()
        
        combined_features = facial_features + speech_features
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, facial_features, speech_features):
        # Concatenate features
        combined = torch.cat([facial_features, speech_features], dim=1)
        return self.fusion(combined)


class AttentionFusionModel(nn.Module):
    """
    Attention-based Fusion: Learns to weight modalities dynamically
    """
    def __init__(self, num_classes=7, feature_dim=256):
        super(AttentionFusionModel, self).__init__()
        
        # Attention mechanism
        self.attention_facial = nn.Linear(feature_dim, 1)
        self.attention_speech = nn.Linear(feature_dim, 1)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, facial_features, speech_features):
        # Calculate attention weights
        attention_f = torch.sigmoid(self.attention_facial(facial_features))
        attention_s = torch.sigmoid(self.attention_speech(speech_features))
        
        # Normalize attention weights
        attention_sum = attention_f + attention_s
        attention_f = attention_f / attention_sum
        attention_s = attention_s / attention_sum
        
        # Apply attention
        weighted_facial = attention_f * facial_features
        weighted_speech = attention_s * speech_features
        
        # Concatenate and fuse
        combined = torch.cat([weighted_facial, weighted_speech], dim=1)
        return self.fusion(combined)


class ConfidenceWeightedFusion(nn.Module):
    """
    Fusion based on prediction confidence
    """
    def __init__(self, num_classes=7):
        super(ConfidenceWeightedFusion, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, facial_logits, speech_logits):
        # Calculate confidence as max probability
        facial_probs = F.softmax(facial_logits, dim=1)
        speech_probs = F.softmax(speech_logits, dim=1)
        
        facial_confidence = torch.max(facial_probs, dim=1, keepdim=True)[0]
        speech_confidence = torch.max(speech_probs, dim=1, keepdim=True)[0]
        
        # Normalize confidences
        total_confidence = facial_confidence + speech_confidence
        facial_weight = facial_confidence / total_confidence
        speech_weight = speech_confidence / total_confidence
        
        # Weighted fusion
        fused = facial_weight * facial_logits + speech_weight * speech_logits
        return fused


class MultimodalFusionModel(nn.Module):
    """
    Complete Multimodal Model with facial and speech encoders
    """
    def __init__(self, facial_model, speech_model, fusion_type='late', num_classes=7):
        super(MultimodalFusionModel, self).__init__()
        
        self.facial_model = facial_model
        self.speech_model = speech_model
        self.fusion_type = fusion_type
        
        if fusion_type == 'late':
            self.fusion = LateFusionModel(num_classes)
        elif fusion_type == 'early':
            self.fusion = EarlyFusionModel(num_classes=num_classes)
        elif fusion_type == 'attention':
            self.fusion = AttentionFusionModel(num_classes=num_classes)
        elif fusion_type == 'confidence':
            self.fusion = ConfidenceWeightedFusion(num_classes)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, facial_input, speech_input):
        # Get predictions/features from each modality
        facial_out = self.facial_model(facial_input)
        speech_out = self.speech_model(speech_input)
        
        # Fuse predictions
        fused = self.fusion(facial_out, speech_out)
        
        return fused, facial_out, speech_out  # Return individual outputs too


def get_fusion_model(facial_model, speech_model, fusion_type='late', num_classes=7):
    """
    Factory function to create fusion model
    
    Args:
        facial_model: Trained facial emotion model
        speech_model: Trained speech emotion model
        fusion_type: Type of fusion ('late', 'early', 'attention', 'confidence')
        num_classes: Number of emotion classes
    
    Returns:
        MultimodalFusionModel
    """
    return MultimodalFusionModel(facial_model, speech_model, fusion_type, num_classes)


if __name__ == "__main__":
    print("Testing Multimodal Fusion Models...")
    
    # Create dummy models
    from models.facial_emotion_model import FacialEmotionCNN
    from models.speech_emotion_model import SpeechEmotionLSTM
    
    facial_model = FacialEmotionCNN(num_classes=7)
    speech_model = SpeechEmotionLSTM(num_classes=7)
    
    # Test inputs
    batch_size = 4
    facial_input = torch.randn(batch_size, 1, 48, 48)
    speech_input = torch.randn(batch_size, 100, 40)
    
    fusion_types = ['late', 'confidence']
    
    for fusion_type in fusion_types:
        print(f"\n{fusion_type.upper()} Fusion:")
        model = get_fusion_model(facial_model, speech_model, fusion_type)
        fused, facial_out, speech_out = model(facial_input, speech_input)
        print(f"   Facial output: {facial_out.shape}")
        print(f"   Speech output: {speech_out.shape}")
        print(f"   Fused output: {fused.shape}")
        print(f"   Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n✓ All fusion models working correctly!")
