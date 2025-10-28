"""
Training Script for Speech Emotion Recognition Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import sys

sys.path.append('..')

from models.speech_emotion_model import get_model
from utils.speech_preprocessing import prepare_speech_dataloader

class SpeechEmotionTrainer:
    def __init__(self, model, device, save_dir='checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.save_dir / 'logs'))
        self.best_val_acc = 0.0
        
    def train_epoch(self, train_loader, criterion, optimizer, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs, learning_rate, weight_decay=1e-4):
        """Complete training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*60}")
            
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, epoch)
            val_loss, val_acc = self.validate(val_loader, criterion, epoch)
            
            scheduler.step(val_acc)
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            self.save_checkpoint(epoch, val_acc, optimizer)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_best_model()
                print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
        
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=4)
        
        self.writer.close()
        return history
    
    def save_checkpoint(self, epoch, val_acc, optimizer):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pth')
    
    def save_best_model(self):
        """Save best model"""
        torch.save(self.model.state_dict(), self.save_dir / 'best_model.pth')
        main_model_dir = Path('../data/models')
        main_model_dir.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), main_model_dir / 'speech_emotion_model.pth')


def main():
    parser = argparse.ArgumentParser(description='Train Speech Emotion Recognition Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to audio dataset')
    parser.add_argument('--dataset_type', type=str, default='ravdess', 
                        choices=['ravdess', 'crema_d', 'tess'],
                        help='Dataset type')
    parser.add_argument('--model_type', type=str, default='lstm', 
                        choices=['lstm', 'cnn', 'cnn_lstm', 'attention'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--n_mfcc', type=int, default=40, help='Number of MFCC coefficients')
    parser.add_argument('--save_dir', type=str, default='checkpoints/speech', help='Directory to save models')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MAITRI - Speech Emotion Recognition Training")
    print("="*60)
    print(f"Dataset: {args.dataset_type}")
    print(f"Model Type: {args.model_type}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"MFCC Coefficients: {args.n_mfcc}")
    print("="*60)
    
    device = torch.device(args.device)
    
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = prepare_speech_dataloader(
        args.data_path,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        n_mfcc=args.n_mfcc
    )
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Val samples: {len(val_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    print(f"\nCreating {args.model_type} model...")
    model = get_model(
        model_type=args.model_type,
        input_size=args.n_mfcc,
        seq_len=args.max_length,
        num_classes=8
    )
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = SpeechEmotionTrainer(model, device, save_dir=args.save_dir)
    
    print("\nStarting training...")
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    test_loss, test_acc = trainer.validate(test_loader, nn.CrossEntropyLoss(), epoch='Test')
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    print("\n✓ Training completed!")
    print(f"✓ Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"✓ Models saved in: {args.save_dir}")


if __name__ == "__main__":
    main()
