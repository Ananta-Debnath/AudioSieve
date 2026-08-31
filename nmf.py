import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import os


def calculate_nmf(spectra, n_components=10, max_iter=500):
    """
    Apply Non-negative Matrix Factorization to an STFT spectrogram.

    Parameters
    ----------
    spectra : np.ndarray
        Complex STFT matrix with shape:
        (time_frames, frequency_bins)

    n_components : int
        Number of NMF components to extract.

    max_iter : int
        Maximum number of NMF iterations.

    Returns
    -------
    W : np.ndarray
        Frequency profiles of components.
        Shape: (frequency_bins, n_components)

    H : np.ndarray
        Time activation of components.
        Shape: (n_components, time_frames)

    reconstructed_magnitude : np.ndarray
        Magnitude spectrogram reconstructed from NMF.
        Shape: (time_frames, frequency_bins)
    """

    # NMF only works with non-negative values
    magnitude = np.abs(spectra)

    # Your spectra are:
    # (time_frames, frequency_bins)
    #
    # For audio NMF, we want:
    # V = (frequency_bins, time_frames)
    V = magnitude.T

    model = NMF(
        n_components=n_components,
        init="nndsvda",
        max_iter=max_iter,
        random_state=0
    )

    # sklearn gives:
    #
    # V ≈ W @ H
    #
    # W: frequency_bins × components
    # H: components × time_frames
    W = model.fit_transform(V)
    H = model.components_

    # Reconstruct magnitude spectrogram
    reconstructed_V = W @ H

    # Convert back to your normal STFT orientation
    reconstructed_magnitude = reconstructed_V.T

    return W, H, reconstructed_magnitude


def reconstruct_spectra(W, H, original_spectra):
    """
    Reconstruct a complex STFT from NMF W/H and the
    phase of the original STFT.

    Parameters
    ----------
    W : np.ndarray
        NMF frequency profiles.
        Shape: (frequency_bins, components)

    H : np.ndarray
        NMF time activations.
        Shape: (components, time_frames)

    original_spectra : np.ndarray
        Original complex STFT.
        Shape: (time_frames, frequency_bins)

    Returns
    -------
    reconstructed_spectra : np.ndarray
        Complex reconstructed STFT.
        Shape: (time_frames, frequency_bins)
    """

    # Reconstruct magnitude
    reconstructed_magnitude = (W @ H).T

    # Get original phase
    phase = np.angle(original_spectra)

    # Reconstruct complex spectra
    reconstructed_spectra = (
        reconstructed_magnitude *
        np.exp(1j * phase)
    )

    return reconstructed_spectra


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

    plt.figure(figsize=(12, 6))

    for k in range(n_components):
        plt.plot(
            frequencies,
            W[:, k],
            label=f"Component {k}"
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Activation")
    plt.title("NMF Frequency Profiles (W)")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "W_frequency_profiles.png"),
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
    # Individual components
    # --------------------------------------------------

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
                output_dir,
                f"component_{k:02d}.png"
            ),
            dpi=150
        )

        plt.close()

    print(f"NMF visualizations saved to: {output_dir}")