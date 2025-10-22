"""
Data Preprocessing for Facial Emotion Recognition
"""

import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class FER2013Dataset(Dataset):
    """
    FER2013 Dataset loader
    """
    def __init__(self, csv_file=None, root_dir=None, transform=None, split='train'):
        """
        Args:
            csv_file: Path to FER2013 CSV file
            root_dir: Path to image directory (if using image files)
            transform: Transformations to apply
            split: 'train', 'val', or 'test'
        """
        self.transform = transform
        self.split = split
        
        # Emotion labels
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
        if csv_file and os.path.exists(csv_file):
            # Load from CSV
            self.data = pd.read_csv(csv_file)
            self.data = self.data[self.data['Usage'] == split]
        elif root_dir and os.path.exists(root_dir):
            # Load from directory structure
            self.load_from_directory(root_dir, split)
        else:
            raise ValueError("Either csv_file or root_dir must be provided")
    
    def load_from_directory(self, root_dir, split):
        """Load images from directory structure"""
        self.image_paths = []
        self.labels = []
        
        split_dir = Path(root_dir) / split
        for emotion_idx, emotion_name in self.emotion_labels.items():
            emotion_dir = split_dir / emotion_name
            if emotion_dir.exists():
                for img_path in emotion_dir.glob('*.png'):
                    self.image_paths.append(str(img_path))
                    self.labels.append(emotion_idx)
    
    def __len__(self):
        if hasattr(self, 'data'):
            return len(self.data)
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if hasattr(self, 'data'):
            # From CSV
            row = self.data.iloc[idx]
            pixels = np.array(row['pixels'].split(), dtype=np.uint8)
            image = pixels.reshape(48, 48)
            label = int(row['emotion'])
        else:
            # From directory
            image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
            label = self.labels[idx]
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CustomImageDataset(Dataset):
    """
    Generic image dataset for emotion recognition
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get all image paths and labels
        self.samples = []
        self.class_to_idx = {}
        
        # Scan directory
        for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if class_dir.is_dir():
                self.class_to_idx[class_dir.name] = idx
                for img_path in class_dir.glob('*.png') + class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_facial_transforms(img_size=48, augment=True):
    """
    Get image transformations for facial emotion data
    
    Args:
        img_size: Target image size
        augment: Whether to apply data augmentation
    
    Returns:
        train_transform, val_transform
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, val_transform


def prepare_fer2013_dataloader(data_path, batch_size=64, num_workers=4):
    """
    Prepare FER2013 dataloaders
    
    Args:
        data_path: Path to FER2013 data
        batch_size: Batch size
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform, val_transform = get_facial_transforms(augment=True)
    
    # Check if CSV or directory structure
    if Path(data_path).is_file() and data_path.endswith('.csv'):
        train_dataset = FER2013Dataset(csv_file=data_path, split='Training', transform=train_transform)
        val_dataset = FER2013Dataset(csv_file=data_path, split='PublicTest', transform=val_transform)
        test_dataset = FER2013Dataset(csv_file=data_path, split='PrivateTest', transform=val_transform)
    else:
        train_dataset = FER2013Dataset(root_dir=data_path, split='train', transform=train_transform)
        val_dataset = FER2013Dataset(root_dir=data_path, split='val', transform=val_transform)
        test_dataset = FER2013Dataset(root_dir=data_path, split='test', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def detect_and_crop_face(image, face_cascade=None):
    """
    Detect and crop face from image
    
    Args:
        image: Input image (numpy array)
        face_cascade: OpenCV face cascade classifier
    
    Returns:
        Cropped face image or None
    """
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Take the largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y:y+h, x:x+w]
        return face
    
    return None


if __name__ == "__main__":
    print("Testing Facial Data Preprocessing...")
    
    # Test transforms
    train_transform, val_transform = get_facial_transforms()
    print("✓ Transforms created")
    
    # Test with dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (48, 48), dtype=np.uint8))
    transformed = train_transform(dummy_image)
    print(f"✓ Transform test: {transformed.shape}")
    
    print("\nNote: To test dataset loading, provide actual data path")
