# Speaker Identification System using MFCC + k-NN & GMM

## Overview

This project implements a Speaker Identification System using classical machine learning techniques and speech signal processing.

The system extracts MFCC (Mel Frequency Cepstral Coefficients) along with Delta features from speech signals and performs speaker classification using:

- k-Nearest Neighbors (k-NN)
- Gaussian Mixture Models (GMM)
- Majority Voting for file-level decision

The models are trained and evaluated on selected speakers from the TIMIT speech corpus.

---

## System Pipeline

1. Audio Preprocessing
   - Normalization
   - Pre-emphasis filtering

2. Feature Extraction
   - MFCC computation
   - Delta coefficient calculation
   - 26-dimensional feature vector

3. Model Training
   - k-NN classifier (k = 5)
   - GMM (8 components per speaker)

4. Testing
   - Frame-level prediction
   - File-level prediction using majority voting

5. Evaluation
   - Frame accuracy
   - File accuracy
   - Confusion matrices
   - Voting impact comparison

---

## Dataset

Dataset: TIMIT Speech Corpus  
Selected Speakers: 5  
Training: 6 audio files per speaker  
Testing: Remaining files  

Note: Update the dataset path inside main.m before running.

---

## Models Implemented

### 1. k-Nearest Neighbors (k-NN)
- Distance-based classification
- Standardized feature scaling
- Majority voting for file-level decision

### 2. Gaussian Mixture Model (GMM)
- One GMM per speaker
- 8 Gaussian components
- Log-likelihood scoring
- Probabilistic classification

---

## Performance Metrics

The system evaluates:

- Frame-Level Accuracy
- File-Level Accuracy
- Confusion Matrix Visualization
- Comparison between frame and file prediction
- Impact of majority voting

---

## Technologies Used

- MATLAB
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- MFCC Feature Extraction
- Probabilistic Modeling

---

## How to Run

1. Install MATLAB with:
   - Signal Processing Toolbox
   - Statistics and Machine Learning Toolbox

2. Update the dataset path in main.m:

   datasetPath = 'your_timit_folder_path';

3. Run:

   main.m

4. The system will:
   - Extract features
   - Train models
   - Test predictions
   - Display confusion matrices
   - Print accuracy results

---

## Applications

- Voice Biometrics
- Speaker Authentication Systems
- Security Access Control
- Forensic Audio Analysis
- Speech-based Identity Recognition

---

## Future Improvements

- Implement Universal Background Model (UBM-GMM)
- Add cross-validation
- Compare with Deep Learning models (CNN / LSTM)
- Convert implementation to Python (Librosa + Scikit-learn)

---
speaker-identification
speech-processing
mfcc
gmm
knn
machine-learning
audio-processing
matlab

