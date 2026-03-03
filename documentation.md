# Speaker Identification System Documentation
------------------------------------------------------------

1. INTRODUCTION

Speaker identification is the task of determining which speaker
produced a given speech signal. This project implements a
text-independent speaker identification system using classical
machine learning methods.

The system compares two models:
- k-Nearest Neighbors (k-NN)
- Gaussian Mixture Models (GMM)

Both models are evaluated at:
- Frame Level
- File Level (using majority voting)

------------------------------------------------------------

2. SYSTEM ARCHITECTURE

The system follows the pipeline below:

Audio Input
   ↓
Preprocessing
   ↓
Feature Extraction (MFCC + Delta)
   ↓
Model Training
   ↓
Prediction
   ↓
Evaluation

------------------------------------------------------------

3. DATA PREPROCESSING

Each audio file undergoes:

1. Normalization:
   Audio is scaled to prevent amplitude variations.

2. Pre-emphasis Filtering:
   A high-pass filter is applied:
       H(z) = 1 - 0.97z^-1
   This boosts high-frequency components to improve MFCC extraction.

------------------------------------------------------------

4. FEATURE EXTRACTION

Features used:
- MFCC (Mel Frequency Cepstral Coefficients)
- Delta coefficients

Final feature vector:
- 13 MFCC coefficients
- 13 Delta coefficients
- Total: 26-dimensional feature vector

These features capture:
- Spectral envelope
- Temporal variations

------------------------------------------------------------

5. MODEL 1: k-NEAREST NEIGHBORS (k-NN)

Type: Distance-based classifier  
Parameter: k = 5  

Process:
- Each frame is classified based on nearest neighbors.
- Euclidean distance metric used.
- Standardization applied before training.

Advantages:
- Simple
- Fast training
- Works well for small datasets

Limitations:
- Sensitive to noise
- Performance depends on feature scaling

------------------------------------------------------------

6. MODEL 2: GAUSSIAN MIXTURE MODEL (GMM)

Type: Probabilistic generative model  
Components: 8 Gaussians per speaker  

Each speaker is modeled using:
    p(x) = Σ (w_i * N(x | μ_i, Σ_i))

Classification:
- Log-likelihood computed for each speaker model
- Highest likelihood selected as prediction

Advantages:
- Captures distribution of speech features
- More robust than k-NN
- Widely used in classical speaker recognition

------------------------------------------------------------

7. MAJORITY VOTING

Frame-level predictions can be noisy.

To improve performance:
- Predictions for all frames in a file are collected.
- The most frequent predicted label is chosen.

This significantly improves file-level accuracy.

------------------------------------------------------------

8. PERFORMANCE METRICS

The system evaluates:

1. Frame-Level Accuracy
2. File-Level Accuracy
3. Confusion Matrices
4. Voting Impact Visualization

Typically:
- File-level accuracy > Frame-level accuracy
- GMM performs better than k-NN

------------------------------------------------------------

9. RESULTS INTERPRETATION

Observations:
- Majority voting increases reliability.
- GMM outperforms k-NN due to probabilistic modeling.
- Frame-level errors are reduced at file-level decision.

------------------------------------------------------------

10. APPLICATIONS

- Voice biometrics
- Speaker authentication
- Secure access systems
- Forensic speaker analysis

------------------------------------------------------------

11. FUTURE WORK

- Universal Background Model (UBM-GMM)
- I-vector or X-vector implementation
- Deep learning models (CNN, LSTM)
- Real-time speaker identification
- Python implementation using Librosa + Scikit-learn

------------------------------------------------------------

END OF DOCUMENT
