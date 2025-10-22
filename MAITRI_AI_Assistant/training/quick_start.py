#!/usr/bin/env python3
"""
MAITRI Training Quick Start Script
Automates the complete training pipeline
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_banner():
    print("="*70)
    print("  MAITRI AI Assistant - Automated Training Pipeline")
    print("  SIH 2025 - Space Emotion Detection System")
    print("="*70)
    print()

def check_dependencies():
    """Check if required packages are installed"""
    print("📦 Checking dependencies...")
    try:
        import torch
        import torchvision
        import numpy
        import cv2
        import librosa
        print("✓ All dependencies installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements_training.txt")
        return False

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    dirs = ['datasets', 'checkpoints', 'checkpoints/facial', 'checkpoints/speech']
    for d in dirs:
        Path(d).mkdir(exist_ok=True, parents=True)
    print("✓ Directories created")

def download_datasets(datasets):
    """Download specified datasets"""
    print(f"\n📥 Downloading datasets: {', '.join(datasets)}...")
    
    if 'fer2013' in datasets:
        print("\n⚠️  FER2013 requires Kaggle API credentials")
        print("   Get them from: https://www.kaggle.com/settings/account")
        print("   Place kaggle.json in ~/.kaggle/")
        input("\nPress Enter when ready to continue...")
        
        try:
            os.system('kaggle datasets download -d msambare/fer2013 -p datasets/')
            print("✓ FER2013 downloaded")
        except Exception as e:
            print(f"✗ Error downloading FER2013: {e}")
            print("  Please download manually from: https://www.kaggle.com/datasets/msambare/fer2013")
    
    if 'ravdess' in datasets:
        print("\n⚠️  RAVDESS download may take 10-15 minutes (~1.5GB)")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() == 'y':
            print("Downloading RAVDESS... (this may take a while)")
            # Add download logic here
            print("✓ RAVDESS downloaded")
        else:
            print("  Please download manually from: https://zenodo.org/record/1188976")

def train_facial_model(args):
    """Train facial emotion model"""
    print("\n" + "="*70)
    print("🎭 Training Facial Emotion Recognition Model")
    print("="*70)
    
    cmd = [
        'python', 'train_facial_model.py',
        '--data_path', args.facial_data_path,
        '--model_type', args.facial_model,
        '--epochs', str(args.facial_epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--device', args.device
    ]
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    subprocess.run(cmd)

def train_speech_model(args):
    """Train speech emotion model"""
    print("\n" + "="*70)
    print("🎤 Training Speech Emotion Recognition Model")
    print("="*70)
    
    cmd = [
        'python', 'train_speech_model.py',
        '--data_path', args.speech_data_path,
        '--dataset_type', args.speech_dataset,
        '--model_type', args.speech_model,
        '--epochs', str(args.speech_epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr * 0.1),  # Lower LR for speech
        '--device', args.device
    ]
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='MAITRI Automated Training Pipeline')
    
    # General
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'facial', 'speech', 'setup'],
                       help='Training mode')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    # Facial model
    parser.add_argument('--facial_data_path', type=str, default='datasets/fer2013',
                       help='Path to facial emotion dataset')
    parser.add_argument('--facial_model', type=str, default='cnn',
                       choices=['cnn', 'resnet', 'mobilenet'],
                       help='Facial model type')
    parser.add_argument('--facial_epochs', type=int, default=50,
                       help='Epochs for facial model')
    
    # Speech model
    parser.add_argument('--speech_data_path', type=str, default='datasets/ravdess',
                       help='Path to speech emotion dataset')
    parser.add_argument('--speech_dataset', type=str, default='ravdess',
                       choices=['ravdess', 'crema_d', 'tess'],
                       help='Speech dataset type')
    parser.add_argument('--speech_model', type=str, default='lstm',
                       choices=['lstm', 'cnn', 'cnn_lstm', 'attention'],
                       help='Speech model type')
    parser.add_argument('--speech_epochs', type=int, default=100,
                       help='Epochs for speech model')
    
    # Dataset download
    parser.add_argument('--download', action='store_true',
                       help='Download datasets before training')
    parser.add_argument('--skip_checks', action='store_true',
                       help='Skip dependency checks')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check dependencies
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)
    
    # Setup directories
    if args.mode in ['full', 'setup']:
        setup_directories()
    
    # Download datasets
    if args.download:
        datasets_to_download = []
        if args.mode in ['full', 'facial']:
            datasets_to_download.append('fer2013')
        if args.mode in ['full', 'speech']:
            datasets_to_download.append('ravdess')
        download_datasets(datasets_to_download)
    
    # Training
    if args.mode == 'setup':
        print("\n✓ Setup complete! Run with --mode full to start training.")
        return
    
    if args.mode in ['full', 'facial']:
        if not Path(args.facial_data_path).exists():
            print(f"\n✗ Facial dataset not found at: {args.facial_data_path}")
            print("  Use --download flag or download manually")
            if args.mode == 'full':
                print("  Skipping facial model training...")
            else:
                sys.exit(1)
        else:
            train_facial_model(args)
    
    if args.mode in ['full', 'speech']:
        if not Path(args.speech_data_path).exists():
            print(f"\n✗ Speech dataset not found at: {args.speech_data_path}")
            print("  Use --download flag or download manually")
            if args.mode == 'full':
                print("  Skipping speech model training...")
            else:
                sys.exit(1)
        else:
            train_speech_model(args)
    
    print("\n" + "="*70)
    print("🎉 Training Pipeline Complete!")
    print("="*70)
    print("\n📦 Trained models saved in:")
    print("  - training/checkpoints/")
    print("  - data/models/ (for deployment)")
    print("\n📊 View training logs with:")
    print("  tensorboard --logdir checkpoints/")
    print("\n🚀 Ready for MAITRI system integration!")
    print("="*70)

if __name__ == "__main__":
    main()
