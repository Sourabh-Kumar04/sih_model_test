# MAITRI AI Assistant - Model Training

Complete training pipeline for Facial and Speech Emotion Recognition models used in the MAITRI AI Assistant.

## 📁 Directory Structure

```
training/
├── datasets/              # Downloaded datasets
│   ├── fer2013/          # Facial emotion dataset
│   ├── ravdess/          # Speech emotion dataset
│   ├── crema_d/          # Additional speech dataset
│   └── tess/             # Additional speech dataset
├── models/               # Model architectures
│   ├── facial_emotion_model.py
│   ├── speech_emotion_model.py
│   └── multimodal_fusion_model.py
├── utils/                # Preprocessing utilities
│   ├── facial_preprocessing.py
│   └── speech_preprocessing.py
├── checkpoints/          # Saved models and logs
├── download_datasets.py  # Dataset downloader
├── train_facial_model.py # Facial model trainer
├── train_speech_model.py # Speech model trainer
└── training_config.yaml  # Configuration file
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements_training.txt
```

### Step 2: Download Datasets

#### Option A: Automated Download (Recommended)

```bash
python download_datasets.py
```

Follow the interactive menu to download datasets.

#### Option B: Manual Download

**FER2013 (Facial Emotion):**
1. Visit: https://www.kaggle.com/datasets/msambare/fer2013
2. Download and extract to `datasets/fer2013/`

**RAVDESS (Speech Emotion):**
1. Visit: https://zenodo.org/record/1188976
2. Download "Audio_Speech_Actors_01-24.zip"
3. Extract to `datasets/ravdess/`

**CREMA-D (Optional):**
1. Visit: https://github.com/CheyneyComputerScience/CREMA-D
2. Download and extract to `datasets/crema_d/`

**TESS (Optional):**
1. Visit: https://tspace.library.utoronto.ca/handle/1807/24487
2. Download and extract to `datasets/tess/`

### Step 3: Configure Kaggle API (for FER2013)

```bash
# Create ~/.kaggle directory
mkdir ~/.kaggle

# Download your kaggle.json from https://www.kaggle.com/settings/account
# Place it in ~/.kaggle/kaggle.json

chmod 600 ~/.kaggle/kaggle.json
```

### Step 4: Train Facial Emotion Model

```bash
python train_facial_model.py \
    --data_path datasets/fer2013 \
    --model_type cnn \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001
```

**Available model types:**
- `cnn` - Custom CNN (lightweight, good for offline)
- `resnet` - ResNet-18 (better accuracy)
- `mobilenet` - MobileNetV2 (most efficient)

### Step 5: Train Speech Emotion Model

```bash
python train_speech_model.py \
    --data_path datasets/ravdess \
    --dataset_type ravdess \
    --model_type lstm \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001
```

**Available model types:**
- `lstm` - Bidirectional LSTM
- `cnn` - 1D CNN
- `cnn_lstm` - Hybrid CNN-LSTM
- `attention` - LSTM with attention

## 📊 Training Arguments

### Facial Emotion Training

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to FER2013 dataset | Required |
| `--model_type` | Model architecture | `cnn` |
| `--epochs` | Number of epochs | `50` |
| `--batch_size` | Batch size | `64` |
| `--lr` | Learning rate | `0.001` |
| `--weight_decay` | L2 regularization | `1e-4` |
| `--save_dir` | Checkpoint directory | `checkpoints/facial` |
| `--device` | Device (cuda/cpu) | Auto-detect |

### Speech Emotion Training

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to audio dataset | Required |
| `--dataset_type` | Dataset type | `ravdess` |
| `--model_type` | Model architecture | `lstm` |
| `--epochs` | Number of epochs | `100` |
| `--batch_size` | Batch size | `32` |
| `--lr` | Learning rate | `0.0001` |
| `--max_length` | Sequence length | `100` |
| `--n_mfcc` | MFCC coefficients | `40` |
| `--save_dir` | Checkpoint directory | `checkpoints/speech` |

## 📈 Expected Results

### Facial Emotion Model

| Model | Accuracy | Size | Training Time (GPU) |
|-------|----------|------|---------------------|
| Custom CNN | 65-70% | ~25MB | ~2 hours |
| ResNet-18 | 70-75% | ~45MB | ~3 hours |
| MobileNetV2 | 68-73% | ~15MB | ~2.5 hours |

### Speech Emotion Model

| Model | Accuracy | Size | Training Time (GPU) |
|-------|----------|------|---------------------|
| LSTM | 75-80% | ~50MB | ~4 hours |
| CNN | 70-75% | ~30MB | ~3 hours |
| CNN-LSTM | 78-82% | ~60MB | ~5 hours |
| Attention | 80-85% | ~55MB | ~5 hours |

## 🔍 Monitoring Training

### TensorBoard

```bash
# View training progress
tensorboard --logdir checkpoints/facial/logs

# Or for speech model
tensorboard --logdir checkpoints/speech/logs
```

Navigate to `http://localhost:6006` to view:
- Training/Validation loss curves
- Accuracy metrics
- Learning rate schedule

## 📦 Model Output

Trained models are saved in two locations:

1. **Training checkpoints:** `training/checkpoints/`
   - All epoch checkpoints
   - Best model
   - Training history (JSON)

2. **Deployment models:** `data/models/`
   - `facial_emotion_model.pth`
   - `speech_emotion_model.pth`
   - Ready for MAITRI system integration

## 🧪 Testing Models

### Test Facial Model

```python
import torch
from models.facial_emotion_model import get_model

# Load model
model = get_model('cnn', num_classes=7)
model.load_state_dict(torch.load('checkpoints/facial/best_model.pth'))
model.eval()

# Test on image
# ... your test code
```

### Test Speech Model

```python
import torch
from models.speech_emotion_model import get_model

# Load model
model = get_model('lstm', num_classes=8)
model.load_state_dict(torch.load('checkpoints/speech/best_model.pth'))
model.eval()

# Test on audio
# ... your test code
```

## 🔧 Troubleshooting

### Out of Memory Error

Reduce batch size:
```bash
python train_facial_model.py --batch_size 32  # Instead of 64
```

### Slow Training

- Use GPU if available
- Reduce `num_workers` if CPU bottleneck
- Use smaller model (mobilenet for facial, cnn for speech)

### Dataset Not Found

Verify dataset structure:
```
datasets/fer2013/
├── train/
│   ├── Angry/
│   ├── Happy/
│   └── ...
├── val/
└── test/
```

### Low Accuracy

- Train for more epochs
- Increase model capacity (resnet instead of cnn)
- Enable data augmentation
- Use multiple datasets (combine RAVDESS + CREMA-D)

## 🎯 Best Practices

1. **Start Small:** Train on subset first to verify pipeline
2. **Monitor Overfitting:** Check train vs val accuracy gap
3. **Data Augmentation:** Essential for small datasets
4. **Learning Rate:** Use ReduceLROnPlateau scheduler
5. **Early Stopping:** Save best validation model, not last epoch
6. **Ensemble:** Combine multiple models for better results

## 🚀 Advanced Usage

### Custom Training Loop

```python
from models.facial_emotion_model import FacialEmotionCNN
from utils.facial_preprocessing import prepare_fer2013_dataloader
import torch.nn as nn
import torch.optim as optim

# Setup
model = FacialEmotionCNN(num_classes=7)
train_loader, val_loader, _ = prepare_fer2013_dataloader('datasets/fer2013')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Multi-GPU Training

```python
# Wrap model with DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to('cuda')
```

### Transfer Learning

```python
# Load pretrained weights
pretrained_model = torch.load('checkpoints/facial/best_model.pth')
model.load_state_dict(pretrained_model)

# Freeze early layers
for param in model.features[:5].parameters():
    param.requires_grad = False

# Fine-tune on new data
# ... training code
```

## 📚 References

### Datasets
- FER2013: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- RAVDESS: [Zenodo](https://zenodo.org/record/1188976)
- CREMA-D: [GitHub](https://github.com/CheyneyComputerScience/CREMA-D)
- TESS: [University of Toronto](https://tspace.library.utoronto.ca/handle/1807/24487)

### Papers
- FER: "Challenges in Representation Learning: A report on three machine learning contests" (2013)
- Speech Emotion: "Speech emotion recognition: Emotional models, databases, features, preprocessing methods, supporting modalities, and classifiers" (2020)

## 🤝 Contributing

To add new datasets or models:

1. Add dataset loader in `utils/`
2. Add model architecture in `models/`
3. Update `training_config.yaml`
4. Update this README

## 📝 License

Part of MAITRI AI Assistant - SIH 2025 Project

## ⚡ Tips for Space Mission Deployment

1. **Model Size:** Use MobileNet/CNN for limited resources
2. **Offline Ready:** All models work without internet
3. **CPU Optimized:** Models run on CPU for spacecraft deployment
4. **Low Latency:** Inference < 100ms per prediction
5. **Robust:** Handles poor lighting and noise
6. **Privacy:** All processing on-device, no data transmission

## 📞 Support

For issues or questions:
- Check troubleshooting section above
- Review training logs in `checkpoints/*/logs/`
- Verify dataset integrity with `download_datasets.py` (option 6)

---

**Happy Training! 🚀**

*Remember: Train on Earth, Deploy in Space!*
