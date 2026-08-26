import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def load_audio(filename, duration=5):
    """Load audio and return sample rate and mono audio."""
    sample_rate, audio = wavfile.read(filename)

    # Convert stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Convert to floating point
    audio = audio.astype(np.float32)

    # Normalize
    max_value = np.max(np.abs(audio))
    if max_value > 0:
        audio /= max_value

    # Keep only the requested duration
    audio = audio[:int(duration * sample_rate)]

    return sample_rate, audio

def save_audio(filename, sample_rate, audio):
    """Save audio as a WAV file."""

    # Prevent clipping
    audio = np.clip(audio, -1.0, 1.0)

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    wavfile.write(
        filename,
        sample_rate,
        audio_int16
    )


def plot_spectrogram(magnitude, sample_rate, hop_size):
    """Display the STFT magnitude as a spectrogram."""

    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    time = np.arange(magnitude.shape[0]) * hop_size / sample_rate
    frequencies = np.linspace(0, sample_rate / 2, magnitude.shape[1])

    plt.figure(figsize=(12, 6))

    plt.imshow(
        magnitude_db.T,
        origin="lower",
        aspect="auto",
        extent=[
            time[0],
            time[-1],
            frequencies[0],
            frequencies[-1]
        ]
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("STFT Spectrogram")
    plt.colorbar(label="Magnitude (dB)")

    plt.show()


def save_spectrogram(magnitude, sample_rate, hop_size, filename):
    """Save the STFT magnitude as a spectrogram image."""

    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    time = np.arange(magnitude.shape[0]) * hop_size / sample_rate
    frequencies = np.linspace(
        0,
        sample_rate / 2,
        magnitude.shape[1]
    )

    plt.figure(figsize=(12, 6))

    plt.imshow(
        magnitude_db.T,
        origin="lower",
        aspect="auto",
        extent=[
            time[0],
            time[-1],
            frequencies[0],
            frequencies[-1]
        ]
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("STFT Spectrogram")
    plt.colorbar(label="Magnitude (dB)")

    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close()


def apply_mask(spectra, mask):
    """Apply a time-frequency mask to the STFT."""

    return spectra * mask


def save_mask(mask, sample_rate, hop_size, filename):
    """Save the binary mask as an image."""

    time = (
        np.arange(mask.shape[0])
        * hop_size
        / sample_rate
    )

    frequencies = np.linspace(
        0,
        sample_rate / 2,
        mask.shape[1]
    )

    plt.figure(figsize=(12, 6))

    plt.imshow(
        mask.T,
        origin="lower",
        aspect="auto",
        extent=[
            time[0],
            time[-1],
            frequencies[0],
            frequencies[-1]
        ]
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Drum Mask")

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()