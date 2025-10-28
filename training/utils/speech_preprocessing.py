"""
Data Preprocessing for Speech Emotion Recognition
"""

import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import soundfile as sf

class SpeechEmotionDataset(Dataset):
    """
    Dataset for Speech Emotion Recognition
    """
    def __init__(self, data_dir, max_length=100, n_mfcc=40, sample_rate=16000, dataset_type='ravdess'):
        """
        Args:
            data_dir: Path to audio files
            max_length: Maximum sequence length (frames)
            n_mfcc: Number of MFCC coefficients
            sample_rate: Audio sample rate
            dataset_type: 'ravdess', 'crema_d', or 'tess'
        """
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.dataset_type = dataset_type
        
        # Emotion mappings for different datasets
        self.emotion_mappings = {
            'ravdess': {
                '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
            },
            'crema_d': {
                'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad',
                'ANG': 'angry', 'FEA': 'fear', 'DIS': 'disgust'
            },
            'tess': {
                'neutral': 'neutral', 'happy': 'happy', 'sad': 'sad',
                'angry': 'angry', 'fear': 'fear', 'disgust': 'disgust', 'ps': 'surprised'
            }
        }
        
        # Standard emotion labels
        self.emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        self.label_to_idx = {label: idx for idx, label in enumerate(self.emotion_labels)}
        
        # Load file paths
        self.audio_files = []
        self.labels = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load audio file paths and labels"""
        if self.dataset_type == 'ravdess':
            self._load_ravdess()
        elif self.dataset_type == 'crema_d':
            self._load_crema_d()
        elif self.dataset_type == 'tess':
            self._load_tess()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def _load_ravdess(self):
        """Load RAVDESS dataset"""
        # RAVDESS filename format: 03-01-06-01-02-01-12.wav
        # Position 3 (index 2) is emotion code
        for audio_file in self.data_dir.rglob('*.wav'):
            parts = audio_file.stem.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                if emotion_code in self.emotion_mappings['ravdess']:
                    emotion = self.emotion_mappings['ravdess'][emotion_code]
                    if emotion in self.label_to_idx:
                        self.audio_files.append(str(audio_file))
                        self.labels.append(self.label_to_idx[emotion])
    
    def _load_crema_d(self):
        """Load CREMA-D dataset"""
        # CREMA-D filename format: 1001_DFA_ANG_XX.wav
        # Position 2 is emotion code
        for audio_file in self.data_dir.rglob('*.wav'):
            parts = audio_file.stem.split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]
                if emotion_code in self.emotion_mappings['crema_d']:
                    emotion = self.emotion_mappings['crema_d'][emotion_code]
                    if emotion in self.label_to_idx:
                        self.audio_files.append(str(audio_file))
                        self.labels.append(self.label_to_idx[emotion])
    
    def _load_tess(self):
        """Load TESS dataset"""
        # TESS directory structure: emotion/filename.wav
        for emotion_dir in self.data_dir.iterdir():
            if emotion_dir.is_dir():
                emotion = emotion_dir.name.lower()
                if emotion in self.emotion_mappings['tess']:
                    emotion_label = self.emotion_mappings['tess'][emotion]
                    if emotion_label in self.label_to_idx:
                        for audio_file in emotion_dir.glob('*.wav'):
                            self.audio_files.append(str(audio_file))
                            self.labels.append(self.label_to_idx[emotion_label])
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Extract features
        features = self.extract_features(audio_path)
        
        return torch.FloatTensor(features), label
    
    def extract_features(self, audio_path):
        """Extract MFCC features from audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            
            # Transpose to (time, features)
            mfcc = mfcc.T
            
            # Pad or truncate to fixed length
            if len(mfcc) < self.max_length:
                pad_width = self.max_length - len(mfcc)
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
            else:
                mfcc = mfcc[:self.max_length, :]
            
            return mfcc
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros((self.max_length, self.n_mfcc))


def extract_audio_features(audio, sr=16000, n_mfcc=40):
    """
    Extract comprehensive audio features
    
    Args:
        audio: Audio time series
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
    
    Returns:
        Dictionary of features
    """
    features = {}
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    features['mfcc'] = mfcc.T
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features['chroma'] = chroma.T
    
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    features['mel'] = mel.T
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features['zcr'] = zcr.T
    
    # RMS Energy
    rms = librosa.feature.rms(y=audio)
    features['rms'] = rms.T
    
    return features


def augment_audio(audio, sr=16000, augment_type='all'):
    """
    Apply audio augmentation
    
    Args:
        audio: Audio time series
        sr: Sample rate
        augment_type: Type of augmentation ('noise', 'pitch', 'speed', 'all')
    
    Returns:
        Augmented audio
    """
    augmented = audio.copy()
    
    if augment_type in ['noise', 'all']:
        # Add white noise
        noise = np.random.randn(len(audio)) * 0.005
        augmented = augmented + noise
    
    if augment_type in ['pitch', 'all']:
        # Pitch shifting
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=np.random.randint(-2, 3))
    
    if augment_type in ['speed', 'all']:
        # Time stretching
        rate = np.random.uniform(0.9, 1.1)
        augmented = librosa.effects.time_stretch(augmented, rate=rate)
    
    return augmented


def prepare_speech_dataloader(data_dir, dataset_type='ravdess', batch_size=32, num_workers=4, 
                               max_length=100, n_mfcc=40):
    """
    Prepare speech emotion dataloaders
    
    Args:
        data_dir: Path to dataset
        dataset_type: Type of dataset
        batch_size: Batch size
        num_workers: Number of workers
        max_length: Maximum sequence length
        n_mfcc: Number of MFCC coefficients
    
    Returns:
        DataLoader
    """
    dataset = SpeechEmotionDataset(
        data_dir=data_dir,
        max_length=max_length,
        n_mfcc=n_mfcc,
        dataset_type=dataset_type
    )
    
    # Split into train/val/test (80/10/10)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing Speech Data Preprocessing...")
    
    # Test feature extraction with dummy audio
    dummy_audio = np.random.randn(16000)  # 1 second at 16kHz
    features = extract_audio_features(dummy_audio)
    print(f"✓ Feature extraction test:")
    for key, value in features.items():
        print(f"   {key}: {value.shape}")
    
    # Test augmentation
    augmented = augment_audio(dummy_audio)
    print(f"\n✓ Augmentation test: {augmented.shape}")
    
    print("\nNote: To test dataset loading, provide actual data path")
