import numpy as np
from pywt import wavedec, waverec
from sklearn.preprocessing import MinMaxScaler

def coefflens(siglen, wavelet, level):
    coefflengths = []
    L = 2
    for i in range(level):
        if i == 0:
            coefflengths.append(siglen // 2)
        elif i == level - 1:
            coefflengths.append(coefflengths[i - 1])
        else:
            coefflengths.append((coefflengths[i - 1] + (L - 1)) // 2)
    coefflengths.reverse()
    return coefflengths

def veclen(siglen, wavelet, level):
    coefflengths = coefflens(siglen, wavelet, level)
    return np.sum(coefflengths)

def sig2coeffs(sig, wavelet, level):
    coeffs = wavedec(sig, wavelet, 'per', level)
    return coeffs

def coeffs2vec(coeffs):
    vec = np.concatenate(coeffs)
    return vec

def coeffs2img(coeffs):
    img = []
    maxlen = coeffs[-1].shape[0]
    for i in range(len(coeffs)):
        coeff_len = coeffs[i].shape[0]
        repeat_count = (maxlen + coeff_len - 1) // coeff_len
        remainder = maxlen % coeff_len
        img_row = np.repeat(coeffs[i], repeat_count)
        if remainder > 0:
            img_row = np.concatenate([img_row, coeffs[i][:remainder]])
        img_row = img_row[:maxlen]
        img.append(img_row)
    return np.stack(img, axis=0)
        
def sig2vec(sig, wavelet):    
    level = np.int32(np.log2(sig.shape[0]))
    coeffs = sig2coeffs(sig, wavelet, level)
    vec = coeffs2vec(coeffs)
    return vec

def sig2img(sig, wavelet):
    level = np.int32(np.log2(sig.shape[0]))
    coeffs = sig2coeffs(sig, wavelet, level)
    img = coeffs2img(coeffs)
    return img

def vec2coeffs(timesteps, vec, wavelet, level):
    coefflengths = coefflens(timesteps, wavelet, level)
    fix_coeffs = []
    for i in range(level + 1):
        fix_coeffs.append(vec[:coefflengths[i]])
        vec = vec[coefflengths[i]:]
    return fix_coeffs

def img2coeffs(timesteps, img, wavelet, level):
    coefflengths = coefflens(timesteps, wavelet, level)
    fix_coeffs = []
    for i in range(level + 1):
        fix_coeffs.append(img[i][:coefflengths[i]])
    return fix_coeffs

def vec2sig(timesteps, vecs, wavelet):
    level = np.int32(np.log2(timesteps))
    coeffs = vec2coeffs(timesteps, vecs, wavelet, level)
    sig = waverec(coeffs, wavelet, 'per')
    return sig

def img2sig(timesteps, img, wavelet):
    level = np.int32(np.log2(timesteps))
    coeffs = img2coeffs(timesteps, img, wavelet, level)
    sig = waverec(coeffs, wavelet, 'per')
    return sig

def multichannel_sig2vec(sig, wavelet):
    vec = []
    for i in range(sig.shape[1]):
        vec.append(sig2vec(sig[:, i], wavelet))
    return np.stack(vec, axis=1)

def multichannel_vec2sig(timesteps, vec, wavelet, channels):
    sig = []
    for i in range(channels):
        sig_n = vec2sig(timesteps, vec[:, i], wavelet)
        sig.append(sig_n)
        
    return np.stack(sig, axis=1)

def multichannel_sig2img(sig, wavelet):
    img = []
    for i in range(sig.shape[1]):
        img.append(sig2img(sig[:, i], wavelet))
    return np.stack(img, axis=2)

def multichannel_img2sig(timesteps, img, wavelet, channels):
    sig = []
    for i in range(channels):
        sig_n = img2sig(timesteps, img[:, i], wavelet)
        sig.append(sig_n)
        
    return np.stack(sig, axis=1)

def batch_multichannel_sig2vec(sig, wavelet):
    vec = []
    for i in range(sig.shape[0]):
        vec.append(multichannel_sig2vec(sig[i], wavelet))
    return np.stack(vec, axis=0)

def batch_multichannel_vec2sig(timesteps, vec, wavelet, channels):
    sig = []
    for i in range(vec.shape[0]):
        sig.append(multichannel_vec2sig(timesteps, vec[i], wavelet, channels))
    return np.stack(sig, axis=0)

def batch_multichannel_sig2img(sig, wavelet):
    img = []
    for i in range(sig.shape[0]):
        img.append(multichannel_sig2img(sig[i], wavelet))
    return np.stack(img, axis=0)

def batch_multichannel_img2sig(timesteps, img, wavelet, channels):
    sig = []
    for i in range(img.shape[0]):
        sig.append(multichannel_img2sig(timesteps, img[i], wavelet, channels))
    return np.stack(sig, axis=0)

def dwtscale(dati_train, dati_val, dati_test):
    # Combine the training and test data
    # combined_data = np.concatenate((dati_train, dati_test), axis=0)
    combined_data = dati_train

    # Reshape combined_data to 2D
    combined_data_reshaped = combined_data.reshape(-1, combined_data.shape[-1])
    
    # Initialize and fit the scaler on the reshaped combined data
    dwtscaler = MinMaxScaler(feature_range=(-1, 1))
    dwtscaler.fit(combined_data_reshaped)
    
    # Reshape dati_train and dati_test to 2D
    dati_train_reshaped = dati_train.reshape(-1, dati_train.shape[-1])
    dati_val_reshaped = dati_val.reshape(-1, dati_val.shape[-1])
    dati_test_reshaped = dati_test.reshape(-1, dati_test.shape[-1])
    
    # Transform the training and test sets separately
    dati_train_scaled_reshaped = dwtscaler.transform(dati_train_reshaped)
    dati_val_scaled_reshaped = dwtscaler.transform(dati_val_reshaped)
    dati_test_scaled_reshaped = dwtscaler.transform(dati_test_reshaped)
    
    # Reshape the scaled data back to their original shape
    dati_train_scaled = dati_train_scaled_reshaped.reshape(dati_train.shape)
    dati_val_scaled = dati_val_scaled_reshaped.reshape(dati_val.shape)
    dati_test_scaled = dati_test_scaled_reshaped.reshape(dati_test.shape)
    
    return dati_train_scaled, dati_val_scaled, dati_test_scaled, dwtscaler

'''
# plot one image in pdf keeping the aspect ratio, the y axis and the x axis basic units must be the same
# i mean that if the x axis long 50 and the y axis long 7 the image must be 50x7

import matplotlib.pyplot as plt

num_samples = 100  # Number of samples in each channel

signal = np.random.rand(num_samples) * 2 - 1  # Random signal between -1 and 1

wavelet = 'haar'

img = sig2img(signal, wavelet)

# plot one image in pdf keeping the aspect ratio, the y axis and the x axis basic units must be the same
# i mean that if the x axis long 50 and the y axis long 7 the image must be 50x7

plt.figure(figsize=(12, 4))
plt.imshow(img, aspect='auto', cmap='viridis')
plt.axis('off')
plt.tight_layout()
plt.savefig('image_representation.pdf')
plt.show()
'''






'''
import numpy as np
import matplotlib.pyplot as plt

# Generate a random multichannel signal (6 channels)
num_samples = 100  # Number of samples in each channel
num_channels = 6
signal = np.random.rand(num_samples, num_channels) * 2 - 1  # Random signal between -1 and 1

wavelet = 'haar'

# Create vector and image representations
vec = multichannel_sig2vec(signal, wavelet)
img = multichannel_sig2img(signal, wavelet)

# Plot and save the original signals
plt.figure(figsize=(12, 8))
for i in range(num_channels):
    plt.subplot(num_channels, 1, i + 1)
    plt.plot(signal[:, i])
    plt.axis('off')
plt.tight_layout()
plt.savefig('original_signals.png')
plt.show()

reshaped_vec = vec.reshape(1, vec.shape[0], vec.shape[1])

# Plot and save the vector representations
plt.figure(figsize=(12, 8))
for i in range(num_channels):
    plt.subplot(num_channels, 1, i + 1)
    plt.imshow(reshaped_vec[0, :, i].reshape(1, -1), aspect='auto', cmap='viridis')
    plt.axis('off')
plt.tight_layout()
plt.savefig('vector_representations.png')
plt.show()

# Plot and save the image representations
plt.figure(figsize=(12, 8))
for i in range(num_channels):
    plt.subplot(num_channels, 1, i + 1)
    plt.imshow(img[:, :, i], aspect='auto', cmap='viridis')
    plt.axis('off')
plt.tight_layout()
plt.savefig('image_representations.png')
plt.show()
'''
