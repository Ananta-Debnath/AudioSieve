import librosa
import soundfile as sf

# 1. Load your original song
y, sr = librosa.load("Audio/trimmed.wav")

# 2. Split into harmonic and percussive audio arrays
y_harmonic, y_percussive = librosa.effects.hpss(y)

# 3. Save them directly to your hard drive
sf.write("Audio/harmonic_only.wav", y_harmonic, sr)
sf.write("Audio/percussive_only.wav", y_percussive, sr)