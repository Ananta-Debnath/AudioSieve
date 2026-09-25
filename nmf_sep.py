import numpy as np
import pandas as pd
from scipy.signal import find_peaks



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



def analyze_vocal_features(
    W,
    H,
    sample_rate,
    n_fft,
    eps=1e-10
):
    """
    Extract a small set of vocal-oriented features
    from NMF components.

    W : (K, N)
        NMF basis matrix.

    H : (N, T)
        NMF activation matrix.

    Returns
    -------
    df : pandas.DataFrame
        One row per NMF component.
    """

    W = np.maximum(W, 0)
    H = np.maximum(H, 0)

    K, num_components = W.shape

    freqs = np.linspace(
        0,
        sample_rate / 2,
        K
    )

    results = []

    for component_idx in range(num_components):

        W_i = W[:, component_idx]
        H_i = H[component_idx]

        # ----------------------------------------------------
        # Reconstructed component
        # ----------------------------------------------------

        component = (
            W_i[:, None] *
            H_i[None, :]
        )

        spectrum = np.mean(
            component,
            axis=1
        )

        spectrum = np.maximum(
            spectrum,
            0
        )

        total_energy = (
            np.sum(spectrum) + eps
        )

        # ====================================================
        # 1. Formant strength
        # ====================================================

        formant_bands = [
            (300, 1000),
            (800, 2500),
            (1800, 3500)
        ]

        formant_values = []

        for low, high in formant_bands:

            mask = (
                (freqs >= low) &
                (freqs <= high)
            )

            if np.any(mask):

                formant_values.append(
                    np.sum(spectrum[mask])
                    / total_energy
                )

        formant_strength = (
            np.mean(formant_values)
            if formant_values
            else 0.0
        )

        # ====================================================
        # 2. Spectral envelope smoothness
        # ====================================================

        if len(spectrum) >= 15:

            kernel = np.ones(15) / 15

            envelope = np.convolve(
                spectrum,
                kernel,
                mode="same"
            )

            envelope_change = np.mean(
                np.abs(np.diff(envelope))
            )

            spectral_envelope_smoothness = (
                1.0 /
                (
                    1.0 +
                    envelope_change /
                    (
                        np.mean(envelope)
                        + eps
                    )
                )
            )

        else:

            spectral_envelope_smoothness = 0.0

        # ====================================================
        # 3. Pitch trajectory
        # ====================================================

        pitch_mask = (
            (freqs >= 70) &
            (freqs <= 500)
        )

        pitch_freqs = freqs[pitch_mask]

        pitch_spectrum = (
            component[pitch_mask]
        )

        if len(pitch_freqs) > 0:

            peak_indices = np.argmax(
                pitch_spectrum,
                axis=0
            )

            pitch = pitch_freqs[
                peak_indices
            ]

            peak_energy = np.max(
                pitch_spectrum,
                axis=0
            )

            frame_energy = (
                np.sum(
                    pitch_spectrum,
                    axis=0
                ) + eps
            )

            pitch_confidence = (
                peak_energy /
                frame_energy
            )

        else:

            pitch = np.zeros(
                component.shape[1]
            )

            pitch_confidence = np.zeros(
                component.shape[1]
            )

        # # ====================================================
        # # 4. Pitch continuity
        # # ====================================================

        # valid_pitch = (
        #     pitch_confidence >= 0.10
        # )

        # if np.sum(valid_pitch) >= 3:

        #     valid_p = pitch[
        #         valid_pitch
        #     ]

        #     log_pitch = np.log2(
        #         np.maximum(
        #             valid_p,
        #             eps
        #         )
        #     )

        #     pitch_changes = np.abs(
        #         np.diff(log_pitch)
        #     )

        #     if len(pitch_changes) > 0:

        #         pitch_continuity = np.exp(
        #             -10.0 *
        #             np.median(
        #                 pitch_changes
        #             )
        #         )

        #     else:

        #         pitch_continuity = 0.0

        # else:

        #     pitch_continuity = 0.0

        # pitch_continuity = float(
        #     np.clip(
        #         pitch_continuity,
        #         0,
        #         1
        #     )
        # )

        # # ====================================================
        # # 5. Pitch range
        # # ====================================================

        # if np.sum(valid_pitch) >= 3:

        #     valid_p = pitch[
        #         valid_pitch
        #     ]

        #     min_pitch = np.min(
        #         valid_p
        #     )

        #     max_pitch = np.max(
        #         valid_p
        #     )

        #     if min_pitch > 0:

        #         pitch_range_semitones = (
        #             12 *
        #             np.log2(
        #                 max_pitch /
        #                 (min_pitch + eps)
        #             )
        #         )

        #     else:

        #         pitch_range_semitones = 0.0

        # else:

        #     pitch_range_semitones = 0.0

        # ---------------------------------------------------------
        # Mid-band ratio (300–4000 Hz)
        # ---------------------------------------------------------

        W_i = W[:, component_idx]
        H_i = H[component_idx]

        component = W_i[:, None] * H_i[None, :]

        spectrum = np.mean(component, axis=1)

        mid_mask = (freqs >= 300) & (freqs <= 4000)

        mid_band_ratio = (
            np.sum(spectrum[mid_mask]) /
            (np.sum(spectrum) + eps)
        )

        # Keep the raw semitone range.
        # Don't convert it to a score yet.
        # It is more useful for analysis.
        
        results.append({
            "component": component_idx,
            "formant_strength":
                float(formant_strength),
            "spectral_envelope_smoothness":
                float(spectral_envelope_smoothness),
            # "pitch_continuity":
            #     float(pitch_continuity),
            # "pitch_range_semitones":
            #     float(pitch_range_semitones),
            "mid_band_ratio":
                float(mid_band_ratio),
        })

    df = pd.DataFrame(results)
    # df.to_csv("nmf_vocal_features.csv", index=False)

    return df


def normalize_column(column):
    min_val = column.min()
    max_val = column.max()

    if max_val == min_val:
        return column * 0.0

    return (column - min_val) / (max_val - min_val)


def score_components(df):
    columns_to_normalize = [
        "energy",
        "spectral_centroid",
        "spectral_bandwidth",
        "peakiness",
        "num_peaks",
    ]
    for col in columns_to_normalize:
        df[col] = normalize_column(df[col])
    
    total_energy = df["energy"].values
    bass_ratio = df["bass_ratio"].values
    low_ratio = df["low_ratio"].values
    high_ratio = df["high_ratio"].values
    spectral_centroid = df["spectral_centroid"].values
    spectral_bandwidth = df["spectral_bandwidth"].values
    spectral_flatness = df["spectral_flatness"].values
    peakiness = df["peakiness"].values
    transientness = df["transientness"].values
    sustain = df["sustain"].values
    activation_variance = df["activation_variance"].values
    harmonicity = df["harmonicity"].values
    num_peaks = df["num_peaks"].values
    rhythmicity = df["rhythmicity"].values

    mid_band_ratio = df["mid_band_ratio"].values
    formant_strength = df["formant_strength"].values
    spectral_envelope_smoothness = df["spectral_envelope_smoothness"].values

    # percussive
    percussion_score = (
        0.40 * transientness +
        0.20 * high_ratio +
        0.15 * spectral_flatness +
        0.10 * (1 - sustain) +
        0.10 * rhythmicity +
        0.05 * activation_variance
    )
    df["percussion_score"] = percussion_score

    bass_score = (
        0.40 * bass_ratio +
        0.20 * low_ratio +
        0.15 * (1 - high_ratio) +
        0.10 * sustain +
        0.10 * harmonicity +
        0.05 * (1 - transientness)
    )
    df["bass_score"] = bass_score

    vocal_score = (
        0.45 * mid_band_ratio +
        0.35 * formant_strength +
        0.20 * spectral_envelope_smoothness
    )
    df["vocal_score"] = vocal_score

    harmonic_score = (
        0.30 * harmonicity +
        0.25 * sustain +
        0.15 * (1 - transientness) +
        0.10 * (1 - bass_ratio) +
        0.10 * spectral_bandwidth +
        0.10 * (1 - spectral_flatness)
    )
    df["harmonic_score"] = harmonic_score
    
    return df


def get_custom_mask(W, H, spectra, comps, power=2):
    # Wiener-style soft mask for selected components
    magnitude = sum(
        np.outer(W[:, k], H[k, :])
        for k in comps
    )

    magnitude = magnitude.T

    eps = 1e-10

    comp_power = magnitude ** power
    total_power = np.abs(spectra) ** int(power)

    mask = comp_power / (total_power + eps)

    mask = np.clip(mask, 0, 1)

    return mask


def get_rest_mask(mask_dict):
    # Initialize rest_mask as an array of ones with the same shape as the first mask
    first_key = next(iter(mask_dict))
    rest_mask = np.ones_like(mask_dict[first_key])

    mask_sum = np.zeros_like(rest_mask)
    for mask in mask_dict.values():
        mask_sum += mask
    
    # Identify indices where the mask sum exceeds 1
    over_one = mask_sum > 1
    over_one = np.ones_like(mask_sum, dtype=bool)  # set all to True

    # print(f"Mask sum exceeds 1: {np.sum(over_one)}")
    # print(f"Mask sum under 1: {np.sum(~over_one)}")
    
    # Scale all masks in the dictionary at those specific indices so they sum to 1
    for key in mask_dict:
        mask_dict[key][over_one] /= mask_sum[over_one]
        
    # Subtract the original mask_sum from rest_mask (which started as 1s).
    # If the sum was > 1, it will become negative, which the clip below will handle.
    rest_mask -= mask_sum

    # Ensure that rest_mask values are clipped between 0 and 1
    rest_mask = np.clip(rest_mask, 0, 1)

    return rest_mask