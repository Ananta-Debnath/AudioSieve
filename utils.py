import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os


def load_audio(filename, duration=None):
    """Load audio and return sample rate and mono audio.
    If duration is None, loads the full audio.
    """
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

    # Keep only the requested duration if specified
    if duration is not None:
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


def calculate_magnitude(spectra):
    """Calculate magnitude of the frequency spectrum."""
    return np.abs(spectra)


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


def save_nmf_visualizations(W, H, sample_rate, n_fft, output_dir="nmf"):
    """
    Save visualizations of NMF results.

    Saves:
        1. W_frequency_profiles.png
        2. H_time_activations.png
        3. component_XX.png for every NMF component
    """

    os.makedirs(output_dir, exist_ok=True)

    n_components = W.shape[1]

    # --------------------------------------------------
    # W: Frequency profiles
    # --------------------------------------------------

    frequencies = np.linspace(
        0,
        sample_rate / 2,
        W.shape[0]
    )

    plt.figure(figsize=(16, 7))

    for k in range(n_components):
        plt.plot(
            frequencies,
            W[:, k],
            label=f"Component {k}"
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Activation")
    plt.title("NMF Frequency Profiles (W)")

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "W_frequency_profiles.png"),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # W: Individual component plots
    # --------------------------------------------------

    individual_dir = os.path.join(output_dir, "W_indiv")
    os.makedirs(individual_dir, exist_ok=True)

    for k in range(n_components):

        plt.figure(figsize=(12, 6))

        plt.plot(
            frequencies,
            W[:, k]
        )

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Activation")
        plt.title(f"NMF Frequency Profile - Component {k}")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                individual_dir,
                f"component_{k:02d}.png"
            ),
            dpi=150
        )

        plt.close()

    # --------------------------------------------------
    # W: Individual component plots
    # --------------------------------------------------

    individual_dir = os.path.join(output_dir, "W_log_indiv")
    os.makedirs(individual_dir, exist_ok=True)

    for k in range(n_components):

        plt.figure(figsize=(12, 6))

        plt.plot(
            frequencies,
            W[:, k]
        )

        plt.xlabel("Frequency (Hz)")
        plt.xscale("symlog", linthresh=1000)
        plt.ylabel("Activation")
        plt.title(f"NMF Frequency Profile - Component {k}")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                individual_dir,
                f"component_{k:02d}.png"
            ),
            dpi=150
        )

        plt.close()


    # --------------------------------------------------
    # H: Time activations
    # --------------------------------------------------

    plt.figure(figsize=(14, 7))

    plt.imshow(
        H,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )

    plt.xlabel("Time Frame")
    plt.ylabel("NMF Component")
    plt.title("NMF Time Activations (H)")
    plt.colorbar(label="Activation")

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "H_time_activations.png"),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # H: Individual component plots
    # --------------------------------------------------

    individual_dir = os.path.join(output_dir, "H_indiv")
    os.makedirs(individual_dir, exist_ok=True)

    for k in range(n_components):
        plt.figure(figsize=(12, 6))

        # Assuming 'times' is your time array and H is shape (n_components, n_frames)
        plt.plot(
            range(H.shape[1]),  # Time frames
            H[k, :] 
        )

        plt.xlabel("Time Frame")
        # Changed y-label to reflect what the y-axis actually represents (magnitude/amplitude)
        plt.ylabel("Activation Amplitude") 
        
        # Added the 'f' prefix here
        plt.title(f"NMF Time Activations (H) - Component {k}")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                individual_dir,
                f"component_{k:02d}.png"
            ),
            dpi=150
        )

        plt.close()


    # --------------------------------------------------
    # Individual components
    # --------------------------------------------------

    individual_dir = os.path.join(output_dir, "V_indiv")
    os.makedirs(individual_dir, exist_ok=True)

    for k in range(n_components):

        component = np.outer(
            W[:, k],
            H[k, :]
        )

        plt.figure(figsize=(14, 6))

        plt.imshow(
            component,
            aspect="auto",
            origin="lower",
            interpolation="nearest"
        )

        plt.xlabel("Time Frame")
        plt.ylabel("Frequency Bin")
        plt.title(f"NMF Component {k}")

        plt.colorbar(label="Magnitude")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                individual_dir,
                f"component_{k:02d}.png"
            ),
            dpi=150
        )

        plt.close()

    print(f"NMF visualizations saved to: {output_dir}")


def nmf_component_masks(W, H, power=2, epsilon=1e-10):
    """
    Generate soft masks for all NMF components.

    Returns
    -------
    masks : np.ndarray
        Shape: (components, time_frames, frequency_bins)
    """

    # W: (frequency, components)
    # H: (components, time)

    components = W[:, :, None] * H[None, :, :]

    # Raise component magnitudes to a power
    components = components ** power

    # Sum across components
    total = np.sum(components, axis=1, keepdims=True)

    # Soft masks
    masks = components / (total + epsilon)

    # Convert:
    # (frequency, components, time)
    # →
    # (components, time, frequency)

    masks = masks.transpose(1, 2, 0)

    return masks


def save_nmf_analysis(df, output_file="Spectograms/nmf_component_analysis.csv"):

    df.to_csv(
        output_file,
        index=False,
        float_format='%.7f'
    )

    print(
        f"\nComponent analysis saved to: {output_file}"
    )