"""
Speech Emotion Recognition Model
LSTM/CNN-based model for audio emotion classification
"""

import torch
import torch.nn as nn

class SpeechEmotionLSTM(nn.Module):
    """
    LSTM-based model for Speech Emotion Recognition
    Input: MFCC features
    Output: Emotion classes
    """
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=8, dropout=0.3):
        super(SpeechEmotionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Pass through FC layers
        out = self.fc(last_output)
        return out


class SpeechEmotionCNN(nn.Module):
    """
    1D CNN for Speech Emotion Recognition
    Processes MFCC features directly
    """
    def __init__(self, input_size=40, seq_len=100, num_classes=8):
        super(SpeechEmotionCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            
            # Conv Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            
            # Conv Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
        )
        
        # Calculate flattened size
        self.flatten_size = 256 * (seq_len // 8)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # Transpose for Conv1d: (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


class SpeechEmotionCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for Speech Emotion Recognition
    CNN extracts local features, LSTM captures temporal patterns
    """
    def __init__(self, input_size=40, seq_len=100, num_classes=8):
        super(SpeechEmotionCNNLSTM, self).__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        
        # CNN feature extraction
        x = self.cnn(x)
        
        # Prepare for LSTM
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        last_output = lstm_out[:, -1, :]
        
        # Classification
        out = self.fc(last_output)
        return out


class SpeechEmotionAttention(nn.Module):
    """
    LSTM with Attention mechanism for Speech Emotion Recognition
    """
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=8):
        super(SpeechEmotionAttention, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        out = self.fc(context)
        return out


def get_model(model_type='lstm', input_size=40, seq_len=100, num_classes=8):
    """
    Factory function to get the appropriate speech emotion model
    
    Args:
        model_type: 'lstm', 'cnn', 'cnn_lstm', or 'attention'
        input_size: Number of MFCC coefficients (default 40)
        seq_len: Sequence length
        num_classes: Number of emotion classes
    
    Returns:
        PyTorch model
    """
    if model_type == 'lstm':
        return SpeechEmotionLSTM(input_size=input_size, num_classes=num_classes)
    elif model_type == 'cnn':
        return SpeechEmotionCNN(input_size=input_size, seq_len=seq_len, num_classes=num_classes)
    elif model_type == 'cnn_lstm':
        return SpeechEmotionCNNLSTM(input_size=input_size, seq_len=seq_len, num_classes=num_classes)
    elif model_type == 'attention':
        return SpeechEmotionAttention(input_size=input_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("Testing Speech Emotion Models...")
    
    batch_size = 4
    seq_len = 100
    input_size = 40
    x = torch.randn(batch_size, seq_len, input_size)
    
    models_to_test = ['lstm', 'cnn', 'cnn_lstm', 'attention']
    
    for model_type in models_to_test:
        print(f"\n{model_type.upper()}:")
        model = get_model(model_type, input_size, seq_len)
        out = model(x)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {out.shape}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n✓ All models working correctly!")
