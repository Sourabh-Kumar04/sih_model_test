"""
Test Script to Verify Training Pipeline Setup
Run this before starting training to ensure everything is configured correctly
"""

import sys
import torch
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import cv2
        import librosa
        import pandas
        import yaml
        from tqdm import tqdm
        print("✓ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Run: pip install -r requirements_training.txt")
        return False

def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU...")
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠ No GPU available, will use CPU (training will be slower)")
        return False

def test_models():
    """Test if model architectures work"""
    print("\nTesting model architectures...")
    
    try:
        sys.path.append('.')
        from models.facial_emotion_model import get_model as get_facial
        from models.speech_emotion_model import get_model as get_speech
        
        # Test facial model
        facial_model = get_facial('cnn')
        x_facial = torch.randn(2, 1, 48, 48)
        out = facial_model(x_facial)
        assert out.shape == (2, 7), f"Facial output shape wrong: {out.shape}"
        print("✓ Facial emotion model working")
        
        # Test speech model
        speech_model = get_speech('lstm')
        x_speech = torch.randn(2, 100, 40)
        out = speech_model(x_speech)
        assert out.shape == (2, 8), f"Speech output shape wrong: {out.shape}"
        print("✓ Speech emotion model working")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing utilities"""
    print("\nTesting preprocessing utilities...")
    
    try:
        from utils.facial_preprocessing import get_facial_transforms
        from utils.speech_preprocessing import extract_audio_features
        
        # Test facial transforms
        train_transform, val_transform = get_facial_transforms()
        print("✓ Facial transforms working")
        
        # Test audio feature extraction
        dummy_audio = np.random.randn(16000)
        features = extract_audio_features(dummy_audio)
        assert 'mfcc' in features, "MFCC not in features"
        print("✓ Audio feature extraction working")
        
        return True
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        return False

def test_directories():
    """Test if required directories exist"""
    print("\nTesting directory structure...")
    
    required_dirs = ['models', 'utils', 'datasets', 'checkpoints']
    all_exist = True
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ {dir_name}/ exists")
        else:
            print(f"⚠ {dir_name}/ missing (will be created)")
            dir_path.mkdir(exist_ok=True)
            all_exist = False
    
    return all_exist

def test_config():
    """Test if configuration file is valid"""
    print("\nTesting configuration...")
    
    try:
        import yaml
        with open('training_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'facial_emotion' in config, "Missing facial_emotion config"
        assert 'speech_emotion' in config, "Missing speech_emotion config"
        print("✓ Configuration file valid")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_dataset_structure():
    """Check if datasets are properly structured"""
    print("\nChecking dataset structure...")
    
    datasets = {
        'FER2013': 'datasets/fer2013',
        'RAVDESS': 'datasets/ravdess'
    }
    
    found_count = 0
    for name, path in datasets.items():
        if Path(path).exists():
            print(f"✓ {name} found at {path}")
            found_count += 1
        else:
            print(f"⚠ {name} not found at {path}")
            print(f"  Download with: python download_datasets.py")
    
    if found_count == 0:
        print("\n⚠ No datasets found. Run: python download_datasets.py")
    
    return found_count > 0

def main():
    print("="*60)
    print("MAITRI Training Pipeline - System Check")
    print("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("GPU Availability", test_gpu),
        ("Model Architectures", test_models),
        ("Preprocessing", test_preprocessing),
        ("Directory Structure", test_directories),
        ("Configuration", test_config),
        ("Dataset Availability", test_dataset_structure)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print("\n" + "="*60)
    print(f"Results: {passed_count}/{total_count} tests passed")
    print("="*60)
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! You're ready to start training!")
        print("\nNext steps:")
        print("  1. Download datasets: python download_datasets.py")
        print("  2. Start training: python quick_start.py --mode full")
    elif passed_count >= total_count - 2:
        print("\n⚠ Most tests passed. Fix minor issues and proceed.")
        print("  Missing datasets can be downloaded later.")
    else:
        print("\n❌ Several tests failed. Please fix issues before training.")
        print("  Check error messages above for details.")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
