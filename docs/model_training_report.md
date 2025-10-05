## **1. Model Overview**

MAITRI uses three specialized neural network models designed for **real-time, offline inference on CPU-constrained devices** (like an embedded system in a spacecraft).

| Model Name                   | Purpose                    | Architecture                           | Key Input Features                                                              | Output                                                                                      |
| ---------------------------- | -------------------------- | -------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **facial_emotion_model.pth** | Facial Emotion Recognition | Lightweight CNN (MobileNetV2 backbone) | 68 Facial Landmarks, Normalized RGB/Grayscale Frames                            | 6-way emotion classification (softmax: e.g., happy, sad, angry, neutral, surprise, disgust) |
| **speech_emotion_model.pth** | Speech Emotion Recognition | 1D CNN + GRU                           | MFCCs (20), Chroma, MelSpectrograms                                             | 6-way emotion classification (softmax)                                                      |
| **multimodal_fusion.pth**    | Final State Prediction     | Simple Feedforward Network (FFN)       | Facial Emotion Confidence Vector, Speech Emotion Confidence Vector, Time Deltas | Fused Emotion Confidence Vector                                                             |

**Notes:**

* The **multimodal fusion model** combines visual and audio cues over time to produce a unified emotional state.
* Low-latency and small memory footprint are critical because the **process_loop runs every second**.

---

## **2. Training Data (Simulated)**

**Facial Data:**

* Base: **FER-2013 dataset** + **custom dataset (200k images)**
* Augmentation: Lighting variation, head pose rotation
* Goal: Robust emotion detection under variable conditions (lighting, camera angles)

**Speech Data:**

* Base datasets: **RAVDESS, TESS, CREMA-D** (~14k utterances)
* Augmentation: Noise addition, simulated bandwidth loss, and noise reduction
* Goal: Accurate emotion recognition under varied acoustic conditions (e.g., spacecraft noise, comm latency)

---

## **3. Optimization and Deployment**

* **Quantization:** All models reduced to **INT8 format** using PyTorch’s quantization (simulated).
* **Size:** Each model < 5MB
* **Reasoning:** Small size + faster inference → suitable for **CPU-only embedded devices**.
* **Process Loop:** Updates every **1 second** to track crew emotional and physical state in near real-time.

**Implications:**

* The system can run **offline**, without relying on cloud servers.
* Fusion of facial + speech emotion ensures **higher confidence in emotional state predictions**.

---


