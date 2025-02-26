# Embedding Models for Multivariate Time Series Anomaly Detection in Industry 5.0

This repository provides the official codebase for the paper "Embedding Models for Multivariate Time Series Anomaly Detection in Industry 5.0".

The repository is comprised of:
- AE.py: defines the Autoencoder class and the losses.
- T2V.py: defines the Time2Vec class anche exposes the methods to create the T2V-based Autoencoder
- dwtutilities.py: exposes the methods to obtain the dwt-based representations
- DWT.py: exposes the methods to create the DWT-based Autoencoders (DWT-based Autoencoder with coefficients as vectors and DWT-based Autoencoder with coefficients as scalograms)