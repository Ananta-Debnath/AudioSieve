import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.signal import find_peaks


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


def analyze_nmf_components(W, H, sample_rate, n_fft):
    """
    Analyze NMF components using signal-processing features.

    Parameters
    ----------
    W : np.ndarray
        NMF frequency profiles.
        Shape: (frequency_bins, components)

    H : np.ndarray
        NMF time activations.
        Shape: (components, time_frames)

    sample_rate : int
        Audio sample rate.

    n_fft : int
        FFT size used for the STFT.

    Returns
    -------
    df : pandas.DataFrame
        Feature table for every NMF component.
    """

    n_freqs, n_components = W.shape

    # --------------------------------------------------
    # Frequency axis
    # --------------------------------------------------

    frequencies = np.linspace(
        0,
        sample_rate / 2,
        n_freqs
    )

    results = []

    for k in range(n_components):

        # --------------------------------------------------
        # Component data
        # --------------------------------------------------

        spectrum = W[:, k].astype(float)
        activation = H[k, :].astype(float)

        # Avoid numerical problems
        spectrum += 1e-12
        activation += 1e-12

        # --------------------------------------------------
        # Normalize frequency profile
        # --------------------------------------------------

        spectrum_norm = spectrum / np.sum(spectrum)

        # --------------------------------------------------
        # Total energy
        # --------------------------------------------------

        total_energy = np.sum(spectrum)

        # --------------------------------------------------
        # Low-frequency energy
        # --------------------------------------------------

        low_mask = frequencies < 250

        bass_mask = frequencies < 150

        high_mask = frequencies > 4000

        low_ratio = (
            np.sum(spectrum[low_mask])
            / total_energy
        )

        bass_ratio = (
            np.sum(spectrum[bass_mask])
            / total_energy
        )

        high_ratio = (
            np.sum(spectrum[high_mask])
            / total_energy
        )

        # --------------------------------------------------
        # Spectral centroid
        # --------------------------------------------------

        centroid = np.sum(
            frequencies * spectrum_norm
        )

        # --------------------------------------------------
        # Spectral bandwidth
        # --------------------------------------------------

        bandwidth = np.sqrt(
            np.sum(
                ((frequencies - centroid) ** 2)
                * spectrum_norm
            )
        )

        # --------------------------------------------------
        # Spectral flatness
        #
        # geometric mean / arithmetic mean
        #
        # close to 1 = noise-like
        # close to 0 = tonal
        # --------------------------------------------------

        geometric_mean = np.exp(
            np.mean(np.log(spectrum))
        )

        arithmetic_mean = np.mean(spectrum)

        flatness = (
            geometric_mean
            / (arithmetic_mean + 1e-12)
        )

        # --------------------------------------------------
        # Spectral peakiness
        # --------------------------------------------------

        peakiness = (
            np.max(spectrum)
            / (np.mean(spectrum) + 1e-12)
        )

        # --------------------------------------------------
        # Activation features
        # --------------------------------------------------

        activation_norm = activation / (
            np.max(activation) + 1e-12
        )

        # Frame-to-frame changes
        activation_diff = np.diff(
            activation_norm
        )

        # Positive changes only
        positive_diff = np.maximum(
            activation_diff,
            0
        )

        # Transientness:
        # strength of the largest activation attacks
        if len(positive_diff) > 0:
            transientness = np.percentile(
                positive_diff,
                95
            )
        else:
            transientness = 0.0

        # Number of significant activation frames
        active_frames = np.sum(
            activation_norm > 0.3
        )

        sustain = (
            active_frames /
            len(activation_norm)
        )

        # Activation variance
        activation_variance = np.var(
            activation_norm
        )

        # --------------------------------------------------
        # Harmonicity approximation
        #
        # Look for multiple strong spectral peaks.
        # Harmonic instruments tend to have several
        # concentrated peaks instead of a flat spectrum.
        # --------------------------------------------------

        # Normalize spectrum
        s = spectrum / np.max(spectrum)

        # Find local peaks
        peaks = []

        for i in range(1, len(s) - 1):

            if (
                s[i] > s[i - 1]
                and s[i] > s[i + 1]
                and s[i] > 0.1
            ):
                peaks.append(i)

        peaks = np.array(peaks)

        if len(peaks) >= 2:

            peak_strength = np.mean(
                s[peaks]
            )

            harmonicity = (
                peak_strength
                * min(len(peaks) / 10, 1.0)
            )

        else:
            harmonicity = 0.0

        # --------------------------------------------------
        # Rythmicity
        # --------------------------------------------------

        rhythmicity = calculate_rhythmicity(activation)

        # --------------------------------------------------
        # Save features
        # --------------------------------------------------

        results.append({

            "component": k,

            "energy": total_energy,

            "bass_ratio": bass_ratio,

            "low_ratio": low_ratio,

            "high_ratio": high_ratio,

            "spectral_centroid": centroid,

            "spectral_bandwidth": bandwidth,

            "spectral_flatness": flatness,

            "peakiness": peakiness,

            "transientness": transientness,

            "sustain": sustain,

            "activation_variance": activation_variance,

            "harmonicity": harmonicity,

            "num_peaks": len(peaks),

            "rhythmicity": rhythmicity
        })

    df = pd.DataFrame(results)

    return df


def rank_nmf_components(df):
    """
    Print useful rankings for NMF components.
    """

    print("\n" + "=" * 60)
    print("NMF COMPONENT ANALYSIS")
    print("=" * 60)

    rankings = {

        "LOW FREQUENCY":
            "low_ratio",

        "BASS":
            "bass_ratio",

        "HIGH FREQUENCY":
            "high_ratio",

        "TRANSIENT":
            "transientness",

        "SUSTAIN":
            "sustain",

        "HARMONIC":
            "harmonicity",

        "TONAL / NON-FLAT":
            "peakiness",

        "NOISE-LIKE":
            "spectral_flatness",

        "SPECTRAL CENTROID":
            "spectral_centroid",

        "ENERGY":
            "energy"
    }

    for name, column in rankings.items():

        print("\n" + "-" * 60)
        print(name)
        print("-" * 60)

        ranked = (
            df
            .sort_values(
                column,
                ascending=False
            )
            [["component", column]]
            .head(10)
        )

        print(
            ranked.to_string(
                index=False
            )
        )



def save_nmf_analysis(df, output_file="Spectograms/nmf_component_analysis.csv"):

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nComponent analysis saved to: {output_file}"
    )


import numpy as np
from scipy.signal import find_peaks


def calculate_rhythmicity(activation):
    """
    Calculate rhythmicity of an NMF component
    using only its temporal activation H[i].

    Returns a score between 0 and 1.
    """

    activation = np.asarray(activation, dtype=float)

    if len(activation) < 3:
        return 0.0

    # Remove mean
    activation = activation - np.mean(activation)

    std = np.std(activation)

    if std < 1e-10:
        return 0.0

    # Normalize
    activation /= std

    # Find significant activation peaks
    peaks, properties = find_peaks(
        activation,
        prominence=0.5,
        distance=2
    )

    if len(peaks) < 3:
        return 0.0

    # Time between consecutive activations
    intervals = np.diff(peaks)

    if len(intervals) < 2:
        return 0.0

    mean_interval = np.mean(intervals)

    if mean_interval < 1e-10:
        return 0.0

    # How consistent are the intervals?
    variation = np.std(intervals) / mean_interval

    regularity = 1.0 / (1.0 + variation)

    # How strong are the peaks?
    peak_strength = np.mean(properties["prominences"])
    peak_strength = peak_strength / (peak_strength + 1.0)

    # Combine
    score = (
        0.7 * regularity +
        0.3 * peak_strength
    )

    return float(np.clip(score, 0.0, 1.0))