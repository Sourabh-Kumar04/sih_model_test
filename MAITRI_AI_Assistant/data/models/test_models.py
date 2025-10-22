"""
MAITRI AI Assistant - Multimodal Emotion Recognition System
PyTorch Implementation for ISRO Crew Health Monitoring

Components:
1. Facial Emotion Recognition (CNN + MobileNetV2)
2. Voice Emotion Recognition (LSTM + Wav2Vec2)
3. Multimodal Fusion Layer
4. Complete Training Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Config

# ============================================
# 1. FACIAL EMOTION RECOGNITION MODEL
# ============================================

class FacialEmotionCNN(nn.Module):
    """
    Facial Emotion Recognition using MobileNetV2 backbone
    Emotions: Neutral, Happy, Sad, Angry, Fear, Surprise, Disgust
    """
    def __init__(self, num_emotions=7, pretrained=True):
        super(FacialEmotionCNN, self).__init__()
        
        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Replace classifier for emotion recognition
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_emotions)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, 224, 224) - RGB facial images
        Returns:
            emotion_logits: (batch_size, num_emotions)
        """
        return self.backbone(x)


# ============================================
# 2. VOICE EMOTION RECOGNITION MODEL
# ============================================

class VoiceEmotionLSTM(nn.Module):
    """
    Voice Emotion Recognition using LSTM + Audio Features
    Features: MFCC, Pitch, Energy, Zero-Crossing Rate
    """
    def __init__(self, input_size=40, hidden_size=256, num_layers=3, num_emotions=7):
        super(VoiceEmotionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM for temporal feature learning
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_emotions)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size) - Audio features
        Returns:
            emotion_logits: (batch_size, num_emotions)
        """
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        # Classification
        output = self.classifier(context)
        return output


class VoiceEmotionWav2Vec2(nn.Module):
    """
    Advanced Voice Emotion Recognition using Wav2Vec2 pretrained model
    Better feature extraction from raw audio
    """
    def __init__(self, num_emotions=7, freeze_feature_extractor=True):
        super(VoiceEmotionWav2Vec2, self).__init__()
        
        # Load pretrained Wav2Vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Optionally freeze feature extractor
        if freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        
        # Emotion classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),  # Wav2Vec2 base output: 768
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_emotions)
        )
        
    def forward(self, input_values):
        """
        Args:
            input_values: (batch_size, sequence_length) - Raw audio waveform
        Returns:
            emotion_logits: (batch_size, num_emotions)
        """
        # Extract features using Wav2Vec2
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # Global average pooling
        pooled = torch.mean(hidden_states, dim=1)  # (batch, 768)
        
        # Classification
        return self.classifier(pooled)


# ============================================
# 3. VITAL SIGNS PROCESSING MODEL
# ============================================

class VitalSignsProcessor(nn.Module):
    """
    Processes vital signs: Heart Rate Variability, Respiration Rate
    Detects stress and cognitive load indicators
    """
    def __init__(self, input_size=10, hidden_size=128, num_states=5):
        super(VitalSignsProcessor, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_states)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_size) - Vital sign features
        Returns:
            state_logits: (batch_size, num_states)
        """
        features = self.encoder(x)
        return self.classifier(features), features


# ============================================
# 4. MULTIMODAL FUSION MODEL
# ============================================

class MAITRIMultimodalFusion(nn.Module):
    """
    Complete MAITRI AI System - Fuses facial, voice, and vital signs
    Provides comprehensive crew health assessment
    """
    def __init__(self, num_emotions=7, num_physical_states=5):
        super(MAITRIMultimodalFusion, self).__init__()
        
        # Individual modality models
        self.facial_model = FacialEmotionCNN(num_emotions=num_emotions)
        self.voice_model = VoiceEmotionLSTM(num_emotions=num_emotions)
        self.vital_model = VitalSignsProcessor(num_states=num_physical_states)
        
        # Fusion layer dimensions
        fusion_input_dim = num_emotions + num_emotions + 128  # face + voice + vitals
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        # Feature projection
        self.facial_proj = nn.Linear(num_emotions, 256)
        self.voice_proj = nn.Linear(num_emotions, 256)
        self.vital_proj = nn.Linear(128, 256)
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        
        # Final classifiers
        self.emotion_classifier = nn.Linear(128, num_emotions)
        self.stress_classifier = nn.Linear(128, 3)  # Low, Medium, High
        self.alert_classifier = nn.Linear(128, 2)   # Normal, Critical
        
    def forward(self, face_img, voice_features, vital_signs):
        """
        Args:
            face_img: (batch, 3, 224, 224) - Facial images
            voice_features: (batch, seq_len, 40) - Audio MFCC features
            vital_signs: (batch, 10) - Vital sign measurements
        Returns:
            dict with emotion, stress level, and alert predictions
        """
        # Extract features from each modality
        face_logits = self.facial_model(face_img)
        voice_logits = self.voice_model(voice_features)
        vital_logits, vital_features = self.vital_model(vital_signs)
        
        # Project to common dimension
        face_proj = self.facial_proj(face_logits).unsqueeze(1)  # (batch, 1, 256)
        voice_proj = self.voice_proj(voice_logits).unsqueeze(1)
        vital_proj = self.vital_proj(vital_features).unsqueeze(1)
        
        # Stack modalities for attention
        modalities = torch.cat([face_proj, voice_proj, vital_proj], dim=1)  # (batch, 3, 256)
        
        # Cross-modal attention
        attended, _ = self.cross_attention(modalities, modalities, modalities)
        
        # Average pooling across modalities
        fused = torch.mean(attended, dim=1)  # (batch, 256)
        
        # Final fusion
        fusion_out = self.fusion_network(fused)
        
        # Multiple predictions
        emotion_pred = self.emotion_classifier(fusion_out)
        stress_pred = self.stress_classifier(fusion_out)
        alert_pred = self.alert_classifier(fusion_out)
        
        return {
            'emotion': emotion_pred,
            'stress_level': stress_pred,
            'alert_status': alert_pred,
            'fusion_features': fusion_out,
            'individual_predictions': {
                'face': face_logits,
                'voice': voice_logits,
                'vitals': vital_logits
            }
        }


# ============================================
# 5. TRAINING UTILITIES
# ============================================

class MAITRITrainer:
    """Training pipeline for MAITRI multimodal system"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Multi-task loss functions
        self.emotion_criterion = nn.CrossEntropyLoss()
        self.stress_criterion = nn.CrossEntropyLoss()
        self.alert_criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer with different learning rates for different components
        self.optimizer = torch.optim.AdamW([
            {'params': model.facial_model.parameters(), 'lr': 1e-4},
            {'params': model.voice_model.parameters(), 'lr': 1e-4},
            {'params': model.vital_model.parameters(), 'lr': 1e-3},
            {'params': model.fusion_network.parameters(), 'lr': 5e-4}
        ], weight_decay=0.01)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
    def train_step(self, face_img, voice_feat, vitals, labels):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(face_img, voice_feat, vitals)
        
        # Compute losses
        emotion_loss = self.emotion_criterion(outputs['emotion'], labels['emotion'])
        stress_loss = self.stress_criterion(outputs['stress_level'], labels['stress'])
        alert_loss = self.alert_criterion(outputs['alert_status'], labels['alert'].float())
        
        # Combined loss with weights
        total_loss = emotion_loss + 0.8 * stress_loss + 1.2 * alert_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'emotion_loss': emotion_loss.item(),
            'stress_loss': stress_loss.item(),
            'alert_loss': alert_loss.item()
        }


# ============================================
# 6. USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Initialize complete MAITRI system
    model = MAITRIMultimodalFusion(num_emotions=7, num_physical_states=5)
    
    # Example inputs
    batch_size = 4
    face_input = torch.randn(batch_size, 3, 224, 224)  # Facial images
    voice_input = torch.randn(batch_size, 100, 40)     # MFCC features
    vital_input = torch.randn(batch_size, 10)          # Vital signs
    
    # Forward pass
    with torch.no_grad():
        predictions = model(face_input, voice_input, vital_input)
    
    print("MAITRI AI Predictions:")
    print(f"Emotion scores: {predictions['emotion'].shape}")
    print(f"Stress level: {predictions['stress_level'].shape}")
    print(f"Alert status: {predictions['alert_status'].shape}")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save model
    torch.save(model.state_dict(), 'maitri_multimodal_model.pth')
    print("\nModel saved successfully!")