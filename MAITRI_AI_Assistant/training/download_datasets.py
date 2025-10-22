"""
Dataset Download Script for MAITRI AI Assistant
Downloads FER2013, RAVDESS, CREMA-D, and TESS datasets
"""

import os
import zipfile
import requests
from pathlib import Path
import kaggle
import gdown

# Base directory for datasets
DATASET_DIR = Path("datasets")
DATASET_DIR.mkdir(exist_ok=True)

def download_fer2013():
    """Download FER2013 dataset from Kaggle"""
    print("Downloading FER2013 dataset...")
    print("Note: You need Kaggle API credentials (~/.kaggle/kaggle.json)")
    print("Get it from: https://www.kaggle.com/settings/account")
    
    try:
        os.system('kaggle datasets download -d msambare/fer2013 -p datasets/')
        
        # Extract
        with zipfile.ZipFile('datasets/fer2013.zip', 'r') as zip_ref:
            zip_ref.extractall('datasets/fer2013')
        
        print("✓ FER2013 downloaded and extracted!")
    except Exception as e:
        print(f"✗ Error downloading FER2013: {e}")
        print("Manual download: https://www.kaggle.com/datasets/msambare/fer2013")

def download_ravdess():
    """Download RAVDESS dataset"""
    print("\nDownloading RAVDESS dataset...")
    
    ravdess_dir = DATASET_DIR / "ravdess"
    ravdess_dir.mkdir(exist_ok=True)
    
    # RAVDESS is split into multiple parts
    urls = [
        "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip",
    ]
    
    for url in urls:
        try:
            filename = url.split('/')[-1]
            filepath = ravdess_dir / filename
            
            if filepath.exists():
                print(f"  {filename} already exists, skipping...")
                continue
            
            print(f"  Downloading {filename}...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r  Progress: {percent:.1f}%", end='')
            
            print(f"\n  Extracting {filename}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(ravdess_dir)
            
            print(f"  ✓ {filename} downloaded and extracted!")
            
        except Exception as e:
            print(f"  ✗ Error downloading RAVDESS: {e}")
            print("  Manual download: https://zenodo.org/record/1188976")

def download_crema_d():
    """Download CREMA-D dataset"""
    print("\nDownloading CREMA-D dataset...")
    print("Note: CREMA-D requires manual download")
    print("Please visit: https://github.com/CheyneyComputerScience/CREMA-D")
    print("Download and extract to: datasets/crema_d/")
    
    crema_dir = DATASET_DIR / "crema_d"
    crema_dir.mkdir(exist_ok=True)

def download_tess():
    """Download TESS dataset"""
    print("\nDownloading TESS dataset...")
    
    tess_dir = DATASET_DIR / "tess"
    tess_dir.mkdir(exist_ok=True)
    
    # TESS Google Drive link (example - you need to update with actual link)
    print("Note: TESS requires manual download")
    print("Please visit: https://tspace.library.utoronto.ca/handle/1807/24487")
    print("Download and extract to: datasets/tess/")

def verify_datasets():
    """Verify that datasets are downloaded correctly"""
    print("\n" + "="*50)
    print("DATASET VERIFICATION")
    print("="*50)
    
    datasets = {
        'FER2013': DATASET_DIR / 'fer2013',
        'RAVDESS': DATASET_DIR / 'ravdess',
        'CREMA-D': DATASET_DIR / 'crema_d',
        'TESS': DATASET_DIR / 'tess'
    }
    
    for name, path in datasets.items():
        if path.exists() and any(path.iterdir()):
            print(f"✓ {name}: Found")
        else:
            print(f"✗ {name}: Not found or empty")

def main():
    print("="*50)
    print("MAITRI AI Assistant - Dataset Downloader")
    print("="*50)
    
    print("\n1. FER2013 (Facial Emotion)")
    print("2. RAVDESS (Speech Emotion - Multimodal)")
    print("3. CREMA-D (Speech Emotion)")
    print("4. TESS (Speech Emotion)")
    print("5. Download All")
    print("6. Verify Datasets")
    print("0. Exit")
    
    choice = input("\nEnter your choice: ")
    
    if choice == '1':
        download_fer2013()
    elif choice == '2':
        download_ravdess()
    elif choice == '3':
        download_crema_d()
    elif choice == '4':
        download_tess()
    elif choice == '5':
        download_fer2013()
        download_ravdess()
        download_crema_d()
        download_tess()
    elif choice == '6':
        verify_datasets()
    elif choice == '0':
        return
    else:
        print("Invalid choice!")
    
    verify_datasets()

if __name__ == "__main__":
    main()
