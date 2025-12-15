import streamlit as st
from pydub import AudioSegment
import io
import librosa
import librosa.display
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import os

st.title("AI audio detect")

uploaded_file = st.file_uploader("Upload audio")
if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[1][1:]  # e.g. 'wav', 'mp3', etc.
    audio = AudioSegment.from_file(io.BytesIO(uploaded_file.read()), format=file_ext)
    
    # Export to WAV into a bytes buffer
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    
    st.success("Conversion complete!")
    
    # Load audio with librosa from the bytes buffer
    signal, sr = librosa.load(wav_io, sr=None, mono=True)
    
    # Display audio info
    st.write(f"Sample Rate: {sr}")
    st.write(f"Signal Length: {len(signal)}")

    # Normalize
    signal = signal / np.max(np.abs(signal))

    # FFT calculation and plotting
    fft_vals = np.fft.fft(signal)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.fftfreq(len(fft_mag), 1/sr)

    half = len(freqs) // 2

    plt.figure(figsize=(10,4))
    plt.plot(freqs[:half], fft_mag[:half])
    plt.title("FFT Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    st.pyplot(plt)
    plt.clf()  # Clear plot before next one

    # STFT calculation and plotting
    stft = librosa.stft(signal, n_fft=1024, hop_length=512, window='hamming')
    stft_mag = np.abs(stft)

    plt.figure(figsize=(10,4))
    librosa.display.specshow(librosa.amplitude_to_db(stft_mag),
                             sr=sr, hop_length=512,
                             x_axis='time', y_axis='hz')
    plt.colorbar(label='dB')
    plt.title("STFT Spectrogram")

    st.pyplot(plt)
    plt.clf()

    # Spectral features
    flatness = np.mean(librosa.feature.spectral_flatness(y=signal))
    centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
    harmonic_energy = np.mean(stft_mag, axis=0)
    harmonic_variation = np.std(harmonic_energy)

    st.write(f"Spectral Flatness: {flatness:.6f}")
    st.write(f"Spectral Centroid: {centroid:.2f} Hz")
    st.write(f"Harmonic Variation (std dev): {harmonic_variation:.6f}")

     # Simple AI vs Human voice detection rule
    if harmonic_variation < 0.5 and flatness < 0.1:
        result = "Human Voice"
    else:
        result = "AI-Generated Voice"

    st.write(f"**Detection Result:** {result}")