#!/usr/bin/env python3
"""
MAITRI Multimodal Fusion Model Training Script
Combines trained facial and speech emotion models for improved accuracy
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multimodal_fusion_model import (
    LateFusionModel, 
    EarlyFusionModel, 
    AttentionFusionModel,
    ConfidenceWeightedFusion,
    MultimodalFusionModel
)
from models.facial_emotion_model import get_model as get_facial_model
from models.speech_emotion_model import get_model as get_speech_model


class MultimodalDataset(Dataset):
    """Dataset that combines facial and speech data for fusion training"""
    def __init__(self, facial_data, speech_data, labels, transform=None):
        assert len(facial_data) == len(speech_data) == len(labels)
        self.facial_data = facial_data
        self.speech_data = speech_data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.facial_data[idx], self.speech_data[idx], self.labels[idx]


def load_pretrained_models(facial_model_path, speech_model_path, 
                          facial_model_type='cnn', speech_model_type='lstm',
                          num_classes=7, device='cuda'):
    """Load pretrained facial and speech models"""
    print(f"Loading pretrained models...")
    
    facial_model = get_facial_model(facial_model_type, num_classes=num_classes)
    if os.path.exists(facial_model_path):
        checkpoint = torch.load(facial_model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            facial_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            facial_model.load_state_dict(checkpoint)
        print(f"✓ Loaded facial model from {facial_model_path}")
    else:
        print(f"⚠ Facial model not found, using random initialization")
    
    speech_model = get_speech_model(speech_model_type, num_classes=num_classes)
    if os.path.exists(speech_model_path):
        checkpoint = torch.load(speech_model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            speech_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            speech_model.load_state_dict(checkpoint)
        print(f"✓ Loaded speech model from {speech_model_path}")
    else:
        print(f"⚠ Speech model not found, using random initialization")
    
    facial_model = facial_model.to(device)
    speech_model = speech_model.to(device)
    
    # Freeze pretrained models
    for param in facial_model.parameters():
        param.requires_grad = False
    for param in speech_model.parameters():
        param.requires_grad = False
    
    facial_model.eval()
    speech_model.eval()
    
    return facial_model, speech_model


def create_fusion_model(fusion_type, facial_model, speech_model, num_classes=7, device='cuda'):
    """Create fusion model based on type"""
    print(f"Creating {fusion_type} fusion model...")
    
    if fusion_type == 'late':
        model = LateFusionModel(num_classes=num_classes)
    elif fusion_type == 'early':
        model = EarlyFusionModel(num_classes=num_classes)
    elif fusion_type == 'attention':
        model = AttentionFusionModel(num_classes=num_classes)
    elif fusion_type == 'confidence':
        model = ConfidenceWeightedFusion(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    if fusion_type in ['late', 'confidence']:
        model = MultimodalFusionModel(
            facial_model, speech_model, fusion_type=fusion_type, num_classes=num_classes
        )
    
    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Fusion model created with {trainable:,} trainable parameters")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, fusion_type):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for facial_input, speech_input, labels in pbar:
        facial_input = facial_input.to(device)
        speech_input = speech_input.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if fusion_type in ['late', 'confidence']:
            outputs, _, _ = model(facial_input, speech_input)
        else:
            outputs = model(facial_input, speech_input)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(dataloader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device, fusion_type):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for facial_input, speech_input, labels in tqdm(dataloader, desc='Validating'):
            facial_input = facial_input.to(device)
            speech_input = speech_input.to(device)
            labels = labels.to(device)
            
            if fusion_type in ['late', 'confidence']:
                outputs, _, _ = model(facial_input, speech_input)
            else:
                outputs = model(facial_input, speech_input)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total


def plot_training_history(history, save_dir):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc, save_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model to {best_path}")


def generate_synthetic_data(num_samples, num_classes, device):
    """Generate synthetic multimodal data for testing"""
    print(f"Generating {num_samples} synthetic samples...")
    facial_data = torch.randn(num_samples, 1, 48, 48)
    speech_data = torch.randn(num_samples, 100, 40)
    labels = torch.randint(0, num_classes, (num_samples,))
    return facial_data, speech_data, labels


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Fusion Model')
    
    # Model paths
    parser.add_argument('--facial_model', type=str, default='checkpoints/facial/best_model.pth')
    parser.add_argument('--speech_model', type=str, default='checkpoints/speech/best_model.pth')
    parser.add_argument('--facial_model_type', type=str, default='cnn', 
                       choices=['cnn', 'resnet', 'mobilenet'])
    parser.add_argument('--speech_model_type', type=str, default='lstm', 
                       choices=['lstm', 'cnn', 'cnn_lstm', 'attention'])
    
    # Fusion configuration
    parser.add_argument('--fusion_type', type=str, default='late', 
                       choices=['late', 'early', 'attention', 'confidence'])
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Data parameters
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--train_samples', type=int, default=1000)
    parser.add_argument('--val_samples', type=int, default=200)
    
    # Saving
    parser.add_argument('--save_dir', type=str, default='checkpoints/fusion')
    parser.add_argument('--device', type=str, default='cuda')
    
    # Fine-tuning
    parser.add_argument('--finetune_all', action='store_true')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained models
    facial_model, speech_model = load_pretrained_models(
        args.facial_model, args.speech_model, args.facial_model_type,
        args.speech_model_type, args.num_classes, device
    )
    
    # Create fusion model
    model = create_fusion_model(
        args.fusion_type, facial_model, speech_model, args.num_classes, device
    )
    
    # Enable fine-tuning if requested
    if args.finetune_all:
        print("Enabling fine-tuning for all layers...")
        for param in model.parameters():
            param.requires_grad = True
    
    # Generate synthetic data (Replace with actual data loading)
    print("\n⚠️  Using synthetic data. Replace with actual multimodal dataset!")
    train_facial, train_speech, train_labels = generate_synthetic_data(
        args.train_samples, args.num_classes, device
    )
    val_facial, val_speech, val_labels = generate_synthetic_data(
        args.val_samples, args.num_classes, device
    )
    
    # Create datasets and dataloaders
    train_dataset = MultimodalDataset(train_facial, train_speech, train_labels)
    val_dataset = MultimodalDataset(val_facial, val_speech, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting Training - {args.fusion_type.upper()} Fusion Model")
    print(f"{'='*70}\n")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.fusion_type
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, args.fusion_type
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        checkpoint_path = os.path.join(args.save_dir, f'fusion_epoch_{epoch+1}.pth')
        save_checkpoint(
            model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc,
            checkpoint_path, is_best
        )
        
        # Save training history
        history_path = os.path.join(args.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
    
    # Plot training history
    plot_training_history(history, args.save_dir)
    
    # Save final model to deployment location
    deployment_path = os.path.join('..', 'data', 'models', 'multimodal_fusion.pth')
    os.makedirs(os.path.dirname(deployment_path), exist_ok=True)
    
    # Load best model
    best_model_path = os.path.join(args.save_dir, f'fusion_epoch_{args.epochs}_best.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save for deployment
    torch.save(model.state_dict(), deployment_path)
    print(f"\n✓ Saved deployment model to {deployment_path}")
    
    # Print final results
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.save_dir}")
    print(f"Deployment model: {deployment_path}")
    print(f"{'='*70}\n")
    
    # Save configuration
    config = vars(args)
    config['best_val_acc'] = best_val_acc
    config['training_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    config_path = os.path.join(args.save_dir, 'fusion_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


if __name__ == '__main__':
    main()
