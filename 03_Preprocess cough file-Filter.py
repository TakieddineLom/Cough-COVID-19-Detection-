import os
import numpy as np
from scipy.signal import butter, filtfilt, decimate
from scipy.io import wavfile


def preprocess_cough(x, fs, cutoff=6000, normalize=True, filter_=True, downsample=True):
    fs_downsample = cutoff * 2
    
    # Preprocess Data
    if len(x.shape) > 1:
        x = np.mean(x, axis=1)  # Convert to mono
    if normalize:
        x = x / (np.max(np.abs(x)) + 1e-17)  # Norm to range between -1 to 1
    if filter_:
        b, a = butter(4, fs_downsample / fs, btype='lowpass')  # 4th order butter lowpass filter
        x = filtfilt(b, a, x)
    if downsample:
        x = decimate(x, int(fs / fs_downsample))  # Downsample for anti-aliasing

    fs_new = fs_downsample

    return np.float32(x), fs_new


def save_preprocessed_cough(filepath, output_dir, cutoff=6000, normalize=True, filter_=True, downsample=True):
    # Load audio
    fs, x = wavfile.read(filepath)

    # Preprocess audio
    x_processed, fs_processed = preprocess_cough(x, fs, cutoff, normalize, filter_, downsample)

    # Save processed audio
    filename = os.path.basename(filepath)
    output_path = os.path.join(output_dir, filename)
    wavfile.write(output_path, fs_processed, x_processed)

# Example usage:
data_folder = "../Covid_19 Project/notebooks/Prob75/"
output_folder = "../Covid_19 Project/Audio Prepro/"
for filename in os.listdir(data_folder):
    if filename.endswith(".wav"):
        input_filepath = os.path.join(data_folder, filename)
        save_preprocessed_cough(input_filepath, output_folder)
