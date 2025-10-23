EXP 1 : ANALYSIS OF DFT WITH AUDIO SIGNAL
AIM:
To analyze audio signal by removing unwanted frequency.

APPARATUS REQUIRED:
PC installed with SCILAB/Python.

PROGRAM:
```
# ==============================
# AUDIO DFT ANALYSIS IN COLAB
# ==============================

# Step 1: Install required packages
!pip install -q librosa soundfile

# Step 2: Upload audio file
from google.colab import files
uploaded = files.upload()   # choose your .wav / .mp3 / .flac file
filename = next(iter(uploaded.keys()))
print("Uploaded:", filename)

# Step 3: Load audio
import librosa, librosa.display
import numpy as np
import soundfile as sf

y, sr = librosa.load(filename, sr=None, mono=True)  # keep original sample rate
duration = len(y) / sr
print(f"Sample rate = {sr} Hz, duration = {duration:.2f} s, samples = {len(y)}")

# Step 4: Play audio
from IPython.display import Audio, display
display(Audio(y, rate=sr))

# Step 5: Full FFT (DFT) analysis
import matplotlib.pyplot as plt

n_fft = 2**14   # choose large power of 2 for smoother spectrum
Y = np.fft.rfft(y, n=n_fft)
freqs = np.fft.rfftfreq(n_fft, 1/sr)
magnitude = np.abs(Y)

plt.figure(figsize=(12,4))
plt.plot(freqs, magnitude)
plt.xlim(0, sr/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Magnitude Spectrum (linear scale)")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
plt.semilogy(freqs, magnitude+1e-12)
plt.xlim(0, sr/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (log scale)")
plt.title("FFT Magnitude Spectrum (log scale)")
plt.grid(True)
plt.show()

# Step 6: Top 10 dominant frequencies
N = 10
idx = np.argsort(magnitude)[-N:][::-1]
print("\nTop 10 Dominant Frequencies:")
for i, k in enumerate(idx):
    print(f"{i+1:2d}. {freqs[k]:8.2f} Hz  (Magnitude = {magnitude[k]:.2e})")

# Step 7: Spectrogram (STFT)
n_fft = 2048
hop_length = n_fft // 4
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(12,5))
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='hz')
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.ylim(0, sr/2)
plt.show()
```
AUDIO USED:
good-morning-242169.mp3

OUTPUT:
Top 10 Dominant Frequencies:

2100.59 Hz (Magnitude = 1.91e+00)
2097.66 Hz (Magnitude = 1.90e+00)
2103.52 Hz (Magnitude = 1.90e+00)
2106.45 Hz (Magnitude = 1.90e+00)
2109.38 Hz (Magnitude = 1.90e+00)
2094.73 Hz (Magnitude = 1.90e+00)
2112.30 Hz (Magnitude = 1.90e+00)
2091.80 Hz (Magnitude = 1.89e+00)
2088.87 Hz (Magnitude = 1.89e+00)
2115.23 Hz (Magnitude = 1.89e+00)
<img width="1010" height="393" alt="image" src="https://github.com/user-attachments/assets/3bb8bcf1-684d-4f74-adde-a145f0f16a84" />
<img width="1012" height="393" alt="image" src="https://github.com/user-attachments/assets/27d8c7b4-a2d2-4ff7-8d6d-f3d1776b4ae5" />
<img width="958" height="470" alt="image" src="https://github.com/user-attachments/assets/426cd1c2-354d-46d2-a143-2d422b2e094f" />

RESULTS
BY USING AUDIO SIGNAL THE DFT IS ANALYSISED USING PYTHON.
