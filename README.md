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

## Repository content
The repository is comprised of:

- [AE.py](AE.py): Defines the `Autoencoder` class and the losses.
  - `Autoencoder` class: Implements an autoencoder with an encoder and decoder.
  - `L2Loss`: Computes the L2 loss between true and predicted values.
  - `Autoencoder.train_step`: Custom training step for the autoencoder.
  - `errors`: Computes DTW, MSE, and L2 errors between true data and predictions.

- [T2V.py](T2V.py): Defines the `Time2Vec` class and exposes the methods to create the T2V-based Autoencoder.
  - `Time2Vec` class: Implements the Time2Vec layer.
  - `create_t2v_encoder`: Creates the encoder part of the T2V-based autoencoder.
  - `create_t2v_decoder`: Creates the decoder part of the T2V-based autoencoder.
  - `create_t2v_autoencoder`: Combines the encoder and decoder to create the T2V-based autoencoder.

- [dwtutilities.py](dwtutilities.py): Exposes the methods to obtain the DWT-based representations.
  - `coefflens`: Computes the lengths of wavelet coefficients.
  - `veclen`: Computes the total length of wavelet coefficients.
  - `sig2coeffs`: Converts a signal to wavelet coefficients.
  - `coeffs2vec`: Converts wavelet coefficients to a vector.
  - `coeffs2img`: Converts wavelet coefficients to an image.
  - `sig2vec`: Converts a signal to a vector using wavelet coefficients.
  - `sig2img`: Converts a signal to an image using wavelet coefficients.
  - `vec2coeffs`: Converts a vector to wavelet coefficients.
  - `img2coeffs`: Converts an image to wavelet coefficients.
  - `vec2sig`: Converts a vector to a signal using wavelet coefficients.
  - `img2sig`: Converts an image to a signal using wavelet coefficients.
  - `multichannel_sig2vec`: Converts a multichannel signal to a vector.
  - `multichannel_vec2sig`: Converts a vector to a multichannel signal.
  - `multichannel_sig2img`: Converts a multichannel signal to an image.
  - `multichannel_img2sig`: Converts an image to a multichannel signal.
  - `batch_multichannel_sig2vec`: Converts a batch of multichannel signals to vectors.
  - `batch_multichannel_vec2sig`: Converts a batch of vectors to multichannel signals.
  - `batch_multichannel_sig2img`: Converts a batch of multichannel signals to images.
  - `batch_multichannel_img2sig`: Converts a batch of images to multichannel signals.
  - `dwtscale`: Scales the DWT coefficients of training, validation, and test data.

- [DWT.py](DWT.py): Exposes the methods to create the DWT-based Autoencoders (DWT-based Autoencoder with coefficients as vectors and DWT-based Autoencoder with coefficients as scalograms).
  - `create_dwt2v_img_encoder`: Creates the encoder part of the DWT-based autoencoder for images.
  - `create_dwt2v_img_decoder`: Creates the decoder part of the DWT-based autoencoder for images.
  - `create_dwt2v_vec_encoder`: Creates the encoder part of the DWT-based autoencoder for vectors.
  - `create_dwt2v_vec_decoder`: Creates the decoder part of the DWT-based autoencoder for vectors.
  - `create_dw2v_vec_autoencoder`: Combines the encoder and decoder to create the DWT-based autoencoder for vectors.
  - `create_dw2v_img_autoencoder`: Combines the encoder and decoder to create the DWT-based autoencoder for images.