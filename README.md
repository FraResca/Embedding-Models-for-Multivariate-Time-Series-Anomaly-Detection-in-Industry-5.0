# Embedding Models for Multivariate Time Series Anomaly Detection in Industry 5.0

This repository provides the official codebase for the paper "Embedding Models for Multivariate Time Series Anomaly Detection in Industry 5.0".

### **1. Time2Vec-based Autoencoder**
- Integrates a custom **Time2Vec (T2V) layer** to capture both periodic and non-periodic temporal patterns.
- Preserves spatial (cross-variable) and temporal dependencies for robust feature extraction.
- Implemented using TensorFlow.

### **2. Discrete Wavelet Transform (DWT)-based Autoencoder**
- Applies **DWT transformation** to preprocess the time series data before encoding.
- Supports two representations:
  - **Flattened DWT coefficients** (1D representation).
  - **Scalogram representation** (2D image-like format).
- Implemented using PyWavelets and Tensorflow.

Both architectures generate embeddings that can be used with various **one-class classification algorithms** for anomaly detection.


The repository is comprised of:
- AE.py: defines the Autoencoder class and the losses.
- T2V.py: defines the Time2Vec class and exposes the methods to create the T2V-based Autoencoder
- dwtutilities.py: exposes the methods to obtain the dwt-based representations
- DWT.py: exposes the methods to create the DWT-based Autoencoders (DWT-based Autoencoder with coefficients as vectors and DWT-based Autoencoder with coefficients as scalograms)
