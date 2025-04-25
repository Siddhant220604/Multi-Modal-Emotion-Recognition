# Multimodal Emotion Recognition (MER)

This project focuses on **Multimodal Emotion Recognition** by combining **Facial Expression Recognition** and **Speech Emotion Recognition** to improve emotion classification accuracy. The system uses a hybrid deep learning approach with **feature-level fusion** of visual and audio modalities.

## 🧠 Project Overview

Emotion recognition is a key aspect of affective computing. While unimodal systems (like facial expression or speech alone) often suffer from ambiguity or low confidence, this project improves robustness and accuracy by fusing features from both modalities.

### Key Components:

- **Facial Emotion Recognition (FER)**:
  - CNN-based model trained on FER2013 dataset.
  - Intermediate feature extraction.
  
- **Speech Emotion Recognition (SER)**:
  - CNN-based model trained on MFCC features extracted from audio.
  - Intermediate feature extraction.

- **Fusion Strategy**:
  - **Feature-Level Fusion**:
    - Global Average Pooling (GAP) applied to both modalities.
    - Feature vectors concatenated and passed to a dense neural network for classification.
      
## 🚀 How It Works

1. **Preprocessing**:
   - Facial images resized and normalized.
   - Audio converted to MFCCs using `librosa`.

2. **Model Training**:
   - Train individual CNN models for FER and SER.
   - Extract intermediate features after convolutional layers.
   - Apply Global Average Pooling.
   - Concatenate features and classify via dense layers.

3. **Prediction**:
   - Input facial image and audio file.
   - Output: Emotion label (e.g., Happy, Sad, Angry, etc.)
