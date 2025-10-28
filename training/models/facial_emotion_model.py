"""
Facial Emotion Recognition Model
Based on ResNet architecture for emotion classification
"""

import torch
import torch.nn as nn
import torchvision.models as models

class FacialEmotionCNN(nn.Module):
    """
    Custom CNN for Facial Emotion Recognition
    Input: 48x48 grayscale images
    Output: 7 emotion classes
    """
    def __init__(self, num_classes=7):
        super(FacialEmotionCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class FacialEmotionResNet(nn.Module):
    """
    ResNet-based model for Facial Emotion Recognition
    Uses transfer learning from ImageNet
    """
    def __init__(self, num_classes=7, pretrained=True):
        super(FacialEmotionResNet, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer to accept grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final layer for emotion classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)


class FacialEmotionMobileNet(nn.Module):
    """
    MobileNetV2-based model for Facial Emotion Recognition
    Lightweight model suitable for offline deployment
    """
    def __init__(self, num_classes=7, pretrained=True):
        super(FacialEmotionMobileNet, self).__init__()
        
        # Load pretrained MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Modify first conv layer for grayscale
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Modify classifier
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.mobilenet(x)


def get_model(model_type='cnn', num_classes=7, pretrained=True):
    """
    Factory function to get the appropriate model
    
    Args:
        model_type: 'cnn', 'resnet', or 'mobilenet'
        num_classes: Number of emotion classes
        pretrained: Use pretrained weights (for resnet/mobilenet)
    
    Returns:
        PyTorch model
    """
    if model_type == 'cnn':
        return FacialEmotionCNN(num_classes)
    elif model_type == 'resnet':
        return FacialEmotionResNet(num_classes, pretrained)
    elif model_type == 'mobilenet':
        return FacialEmotionMobileNet(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("Testing Facial Emotion Models...")
    
    batch_size = 4
    x = torch.randn(batch_size, 1, 48, 48)  # Grayscale 48x48 images
    
    # Test CNN
    print("\n1. Custom CNN:")
    model_cnn = FacialEmotionCNN()
    out = model_cnn(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")
    
    # Test ResNet
    print("\n2. ResNet-18:")
    model_resnet = FacialEmotionResNet(pretrained=False)
    out = model_resnet(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_resnet.parameters()):,}")
    
    # Test MobileNet
    print("\n3. MobileNetV2:")
    model_mobile = FacialEmotionMobileNet(pretrained=False)
    out = model_mobile(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_mobile.parameters()):,}")
    
    print("\n✓ All models working correctly!")
