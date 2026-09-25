import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def initialize_B(K, num_components, random_scale=1.0, seed=None):
    """
    Initialize the NMF basis matrix B with a mixture of
    manually designed spectral profiles and random components.

    Parameters
    ----------
    K : int
        Number of frequency bins.

    num_components : int
        Number of NMF components (J).

    random_scale : float, optional
        Scale of the random components.

    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    B : np.ndarray
        Basis matrix of shape (K, num_components).

    Notes
    -----
    The manually initialized components are broad spectral
    archetypes rather than strict instrument templates.

    The first components are:

        0 : broad/full-spectrum percussion
        1 : low-frequency / kick-like percussion
        2 : high-frequency / snare-like percussion
        3 : deep bass
        4 : broader bass
        5 : low/mid harmonic
        6 : mid/high harmonic

    Remaining components are randomly initialized.
    """

    rng = np.random.default_rng(seed)

    J = num_components
    eps = 1e-12

    # Start everything randomly.
    B = np.abs(rng.standard_normal((K, J))) * random_scale + eps

    # Frequency axis normalized from 0 to 1.
    f = np.linspace(0.0, 1.0, K)

    # ---------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------

    def gaussian(center, width):
        return np.exp(-0.5 * ((f - center) / width) ** 2)

    def lowpass(cutoff, softness):
        return 1.0 / (1.0 + np.exp((f - cutoff) / softness))

    def highpass(cutoff, softness):
        return 1.0 / (1.0 + np.exp(-(f - cutoff) / softness))

    # Number of manually designed components.
    manual_profiles = []

    # ---------------------------------------------------------
    # 1. Broad percussion
    # ---------------------------------------------------------

    # 1. Broadband beat
    broad_percussion = (
        0.70
        + 0.15 * gaussian(0.20, 0.25)
        + 0.15 * gaussian(0.70, 0.25)
    )

    # 2. Low-heavy transient
    low_percussion = (
        0.20
        + 0.80 * gaussian(0.12, 0.16)
    )

    # 3. Mid/high transient
    high_percussion = (
        0.15
        + 0.85 * gaussian(0.65, 0.25)
    )

    # manual_profiles.append(broad_percussion)
    manual_profiles.append(low_percussion)
    manual_profiles.append(high_percussion)

    # ---------------------------------------------------------
    # 2. Low-frequency / kick-like percussion
    # ---------------------------------------------------------

    kick_like = (
        0.10
        + 0.90 * gaussian(0.10, 0.08)
        + 0.25 * gaussian(0.25, 0.12)
    )

    manual_profiles.append(kick_like)

    # ---------------------------------------------------------
    # 3. Mid/high-frequency percussion
    # ---------------------------------------------------------

    snare_like = (
        0.08
        + 0.45 * gaussian(0.35, 0.12)
        + 0.75 * gaussian(0.65, 0.18)
        + 0.35 * gaussian(0.90, 0.10)
    )

    manual_profiles.append(snare_like)

    # ---------------------------------------------------------
    # 4. Deep bass
    # ---------------------------------------------------------

    deep_bass = (
        0.05
        + 1.00 * lowpass(0.16, 0.035)
    )

    manual_profiles.append(deep_bass)

    # ---------------------------------------------------------
    # 5. Broader bass
    # ---------------------------------------------------------

    broad_bass = (
        0.05
        + 0.85 * gaussian(0.12, 0.10)
        + 0.45 * gaussian(0.28, 0.12)
    )

    manual_profiles.append(broad_bass)

    # ---------------------------------------------------------
    # 6. Low/mid harmonic
    # ---------------------------------------------------------

    low_harmonic = (
        0.08
        + 0.50 * gaussian(0.22, 0.16)
        + 0.70 * gaussian(0.42, 0.16)
    )

    manual_profiles.append(low_harmonic)

    # ---------------------------------------------------------
    # 7. Mid/high harmonic
    # ---------------------------------------------------------

    high_harmonic = (
        0.08
        + 0.60 * gaussian(0.50, 0.18)
        + 0.70 * gaussian(0.75, 0.20)
    )

    manual_profiles.append(high_harmonic)

    # ---------------------------------------------------------
    # Insert manual profiles
    # ---------------------------------------------------------

    num_manual = min(len(manual_profiles), J)

    for j in range(num_manual):
        B[:, j] = manual_profiles[j]

    # ---------------------------------------------------------
    # Normalize each component independently.
    #
    # This prevents a manually initialized component from
    # dominating simply because its numerical scale is larger.
    # ---------------------------------------------------------

    B /= np.maximum(np.max(B, axis=0, keepdims=True), eps)

    return B


def visualize_B_diff(B_init, B, output_dir="Spectograms/nmf/B_diff"):
    """
    Create overlapping plot for B initialization and final B after NMF.

    """
    os.makedirs(output_dir, exist_ok=True)

    K, J = B.shape

    for j in range(J):
        plt.figure(figsize=(10, 6))
        plt.plot(B_init[:, j], label="B_init", color="blue", alpha=0.7)
        plt.plot(B[:, j], label="B_final", color="orange", alpha=0.7)
        plt.title(f"Component {j}")
        plt.xlabel("Frequency Bin (K)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True)

        filename = f"B_diff_component_{j:02d}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()


def separate_sources_nmf(X, num_components, alpha=100.0, beta=0.0, max_iter=300, tol=1e-4):
    """
    Separates a magnitude spectrogram X into Basis (B) and Gain (G) matrices 
    using Nonnegative Matrix Factorization with temporal continuity and sparseness.
    
    Parameters:
    X : numpy.ndarray (K, T) - The input magnitude spectrogram (K frequencies, T frames).
    num_components : int (J) - The number of components/basis functions to estimate.
    alpha : float - Weight for the temporal continuity criterion (paper default ~100).
    beta : float - Weight for the sparseness criterion (paper default 0).
    max_iter : int - Maximum number of multiplicative update iterations.
    tol : float - Convergence tolerance threshold.
    
    Returns:
    B : numpy.ndarray (K, J) - The estimated basis matrix (magnitude spectra).
    G : numpy.ndarray (J, T) - The estimated gain matrix (time-varying gains).
    """
    K, T = X.shape
    J = num_components
    eps = 1e-12  # Small constant to prevent division by zero in the pipeline
    
    # 1. Initialize B and G with the absolute value of Gaussian noise

    B = np.abs(np.random.randn(K, J)) + eps
    # B = initialize_B(
    #     K=X.shape[0],
    #     num_components=num_components,
    #     seed=42
    # )
    G = np.abs(np.random.randn(J, T)) + eps
    ones_KT = np.ones((K, T))

    B_init = B.copy()  # Keep a copy of the initial B for visualization
    
    prev_cost = float('inf')
    
    for iteration in range(max_iter):
        # --- UPDATE BASIS MATRIX (B) ---
        X_hat = B @ G + eps
        B_numerator = (X / X_hat) @ G.T
        B_denominator = ones_KT @ G.T + eps
        
        B = B * (B_numerator / B_denominator)
        B = np.maximum(B, eps)  # Enforce strict non-negativity
        
        # --- UPDATE GAIN MATRIX (G) ---
        X_hat = B @ G + eps
        
        # Reconstruction Error Gradients
        grad_cr_pos = B.T @ ones_KT
        grad_cr_neg = B.T @ (X / X_hat)
        
        # Temporal Continuity Gradients
        grad_ct_pos = np.zeros((J, T))
        grad_ct_neg = np.zeros((J, T))
        
        if alpha > 0:
            S2 = np.sum(G**2, axis=1, keepdims=True) + eps
            
            # G_{j, t-1} and G_{j, t+1} with zero-padding at boundaries
            G_prev = np.zeros_like(G)
            G_prev[:, 1:] = G[:, :-1]
            G_next = np.zeros_like(G)
            G_next[:, :-1] = G[:, 1:]
            
            # Sum of squared differences: sum_{i=2}^T (g_{j,i} - g_{j,i-1})^2
            diff_sq_sum = np.sum((G[:, 1:] - G[:, :-1])**2, axis=1, keepdims=True)
            
            grad_ct_pos = (4 * T * G) / S2
            term1 = (2 * T * (G_prev + G_next)) / S2
            term2 = (2 * T * G * diff_sq_sum) / (S2**2)
            grad_ct_neg = term1 + term2
            
        # Sparseness Gradients
        grad_cs_pos = np.zeros((J, T))
        grad_cs_neg = np.zeros((J, T))
        
        if beta > 0:
            S2 = np.sum(G**2, axis=1, keepdims=True) + eps
            S1 = np.sum(G, axis=1, keepdims=True)
            
            grad_cs_pos = 1.0 / np.sqrt((1.0 / T) * S2)
            grad_cs_neg = (G * np.sqrt(T) * S1) / (S2**(1.5))
        
        # Aggregate Positive and Negative Gradients
        grad_c_pos = grad_cr_pos + (alpha * grad_ct_pos) + (beta * grad_cs_pos) + eps
        grad_c_neg = grad_cr_neg + (alpha * grad_ct_neg) + (beta * grad_cs_neg)
        
        # Multiplicative Update for G
        G = G * (grad_c_neg / grad_c_pos)
        G = np.maximum(G, eps)

        # Optional: Calculate convergence cost here using divergence, temporal, and sparseness costs
        # (Omitted for loop speed, assuming fixed max_iter for standard audio processing workflows)

    # Visualize the difference between initial and final B
    visualize_B_diff(B_init, B)

    score = evaluate_decomposition(X, B, G, verbose=True)
    print(f"Final decomposition score: {score:.6f}")
    print()

    return B, G


def save_component_spectrograms(B, G, output_dir="component_spectrograms"):
    """
    Reconstructs and saves each separated component as a spectrogram image.
    
    Parameters:
    B : numpy.ndarray (K, J) - The estimated basis matrix.
    G : numpy.ndarray (J, T) - The estimated gain matrix.
    output_dir : str - The directory to save the output PNG files.
    """
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    K, J = B.shape
    
    for j in range(J):
        # 1. Reconstruct the component's magnitude spectrogram
        # Outer product of the j-th column of B and j-th row of G
        component_spectrogram = np.outer(B[:, j], G[j, :])
        
        # 2. Convert to decibels (dB) for standard audio visualization
        eps = 1e-12 # Prevent log(0)
        component_spectrogram_db = 20 * np.log10(component_spectrogram + eps)
        
        # 3. Plot the spectrogram
        plt.figure(figsize=(10, 6))
        
        # 'origin=lower' ensures low frequencies are at the bottom of the y-axis
        plt.imshow(component_spectrogram_db, aspect='auto', origin='lower', cmap='magma')
        
        plt.title(f"Separated Component {j}")
        plt.ylabel("Frequency Bins (K)")
        plt.xlabel("Time Frames (T)")
        plt.colorbar(label="Magnitude (dB)")
        
        # 4. Save the figure to disk
        filename = f"component_{j:02d}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        
        # Close the figure to free up memory before the next iteration
        plt.close()
        
    print(f"Successfully saved {J} spectrograms to the '{output_dir}' directory.")
    print()

# Example usage (assuming B and G were returned from the previous NMF function):
# save_component_spectrograms(B, G)


def analyze_nmf_components(
    W,
    H,
    sample_rate,
    n_fft,
    eps=1e-10
):
    """
    Analyze NMF components using the actual component spectrogram:

        component(f, t) = W(f, k) * H(k, t)

    Expected shapes:
        W = (frequency_bins, components)
        H = (components, time_frames)

    Returns:
        DataFrame with one row per NMF component.
    """

    n_freqs, n_components = W.shape
    h_components, n_frames = H.shape

    if n_components != h_components:
        raise ValueError(
            f"W and H component counts do not match: "
            f"W={W.shape}, H={H.shape}"
        )

    # Frequency axis
    freqs = np.linspace(
        0,
        sample_rate / 2,
        n_freqs
    )

    results = []

    for k in range(n_components):

        # ---------------------------------------------------------
        # ACTUAL NMF COMPONENT
        # ---------------------------------------------------------
        #
        # W[:, k]       -> (frequency,)
        # H[k, :]       -> (time,)
        #
        # Add dimensions so multiplication gives:
        # (frequency, time)
        #
        component = (
            W[:, k, None] *
            H[k, None, :]
        )

        # ---------------------------------------------------------
        # TOTAL ENERGY
        # ---------------------------------------------------------

        energy = np.sum(component)

        # Energy per time frame
        energy_t = np.sum(component, axis=0)

        # Avoid completely dead components
        total_energy = np.sum(energy_t) + eps

        # ---------------------------------------------------------
        # NORMALIZED COMPONENT
        # ---------------------------------------------------------

        # Normalize each time frame so spectral features describe
        # the shape of the spectrum rather than simply its loudness.
        frame_energy = (
            np.sum(component, axis=0, keepdims=True) + eps
        )

        normalized = component / frame_energy

        # ---------------------------------------------------------
        # SPECTRAL CENTROID
        # ---------------------------------------------------------

        centroid_t = np.sum(
            normalized * freqs[:, None],
            axis=0
        )

        centroid = np.average(
            centroid_t,
            weights=energy_t + eps
        )

        # ---------------------------------------------------------
        # SPECTRAL BANDWIDTH
        # ---------------------------------------------------------

        bandwidth_t = np.sqrt(
            np.sum(
                normalized *
                (freqs[:, None] - centroid_t[None, :]) ** 2,
                axis=0
            )
        )

        bandwidth = np.average(
            bandwidth_t,
            weights=energy_t + eps
        )

        # ---------------------------------------------------------
        # SPECTRAL FLATNESS
        # ---------------------------------------------------------

        geometric_mean = np.exp(
            np.mean(
                np.log(normalized + eps),
                axis=0
            )
        )

        arithmetic_mean = np.mean(
            normalized,
            axis=0
        ) + eps

        flatness_t = geometric_mean / arithmetic_mean

        spectral_flatness = np.average(
            flatness_t,
            weights=energy_t + eps
        )

        # ---------------------------------------------------------
        # PEAKINESS / CREST
        # ---------------------------------------------------------

        peak_t = np.max(normalized, axis=0)

        peakiness_t = (
            peak_t /
            (np.mean(normalized, axis=0) + eps)
        )

        peakiness = np.average(
            peakiness_t,
            weights=energy_t + eps
        )

        # ---------------------------------------------------------
        # LOW / BASS / HIGH ENERGY
        # ---------------------------------------------------------

        bass_mask = freqs < 150
        low_mask = freqs < 250
        high_mask = freqs > 4000

        bass_energy = np.sum(
            component[bass_mask, :]
        )

        low_energy = np.sum(
            component[low_mask, :]
        )

        high_energy = np.sum(
            component[high_mask, :]
        )

        bass_ratio = bass_energy / total_energy
        low_ratio = low_energy / total_energy
        high_ratio = high_energy / total_energy

        # ---------------------------------------------------------
        # SPECTRAL FLUX
        # ---------------------------------------------------------
        #
        # Compare consecutive normalized spectra.
        #
        # Positive flux is especially useful for detecting
        # sudden spectral changes / attacks.
        # ---------------------------------------------------------

        if n_frames > 1:

            diff = np.diff(
                normalized,
                axis=1
            )

            positive_diff = np.maximum(
                diff,
                0
            )

            flux_t = np.sqrt(
                np.sum(
                    positive_diff ** 2,
                    axis=0
                )
            )

            spectral_flux = np.mean(flux_t)
            flux_p90 = np.percentile(flux_t, 90)
            flux_max = np.max(flux_t)

        else:

            spectral_flux = 0.0
            flux_p90 = 0.0
            flux_max = 0.0

        # ---------------------------------------------------------
        # TEMPORAL ACTIVITY
        # ---------------------------------------------------------

        h = H[k, :]

        h_max = np.max(h)

        if h_max > eps:
            h_norm = h / h_max
        else:
            h_norm = np.zeros_like(h)

        # How often is the component substantially active?
        active_ratio = np.mean(
            h_norm > 0.3
        )

        # How often is it almost silent?
        inactive_ratio = np.mean(
            h_norm < 0.05
        )

        # ---------------------------------------------------------
        # TEMPORAL VARIANCE
        # ---------------------------------------------------------

        activation_variance = np.var(
            h_norm
        )

        # ---------------------------------------------------------
        # TEMPORAL BURSTINESS
        # ---------------------------------------------------------

        if len(h_norm) > 1:

            h_diff = np.diff(h_norm)

            positive_h_diff = np.maximum(
                h_diff,
                0
            )

            transientness = np.mean(
                positive_h_diff
            )

            transient_p90 = np.percentile(
                positive_h_diff,
                90
            )

        else:

            transientness = 0.0
            transient_p90 = 0.0

        # ---------------------------------------------------------
        # NUMBER OF SIGNIFICANT ENERGY BURSTS
        # ---------------------------------------------------------

        energy_norm = (
            energy_t /
            (np.max(energy_t) + eps)
        )

        active_energy = energy_norm > 0.3

        # Count starts of active regions
        if len(active_energy) > 1:

            burst_starts = np.sum(
                active_energy[1:] &
                ~active_energy[:-1]
            )

        else:

            burst_starts = int(active_energy[0])

        # ---------------------------------------------------------
        # SUSTAIN
        # ---------------------------------------------------------

        sustain = np.mean(
            energy_norm > 0.3
        )

        # ---------------------------------------------------------
        # SPECTRAL ENERGY DISTRIBUTION
        # ---------------------------------------------------------

        # Weighted average frequency
        # calculated from actual W*H energy.
        weighted_centroid = (
            np.sum(component * freqs[:, None])
            / total_energy
        )

        # ---------------------------------------------------------
        # SAVE
        # ---------------------------------------------------------

        results.append({
            "component": k,

            # -------------------------
            # actual component energy
            # -------------------------
            "energy": energy,

            # -------------------------
            # spectral features
            # -------------------------
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_flatness": spectral_flatness,
            "peakiness": peakiness,

            # -------------------------
            # frequency regions
            # -------------------------
            "bass_ratio": bass_ratio,
            "low_ratio": low_ratio,
            "high_ratio": high_ratio,

            # -------------------------
            # actual spectral movement
            # -------------------------
            "spectral_flux": spectral_flux,
            "flux_p90": flux_p90,
            "flux_max": flux_max,

            # -------------------------
            # temporal behavior
            # -------------------------
            "active_ratio": active_ratio,
            "inactive_ratio": inactive_ratio,
            "activation_variance": activation_variance,

            "transientness": transientness,
            "transient_p90": transient_p90,

            "sustain": sustain,
            "burst_count": burst_starts,

            # -------------------------
            # extra
            # -------------------------
            "weighted_centroid": weighted_centroid
        })

    df = pd.DataFrame(results)
    df_vocal = analyze_vocal_features(
        W, H,
        sample_rate=sample_rate,
        n_fft=n_fft,
        eps=eps
    )

    df_merged = df.merge(
        df_vocal,
        on="component",
        how="left"
    )

    return df_merged


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


def get_framewise_vocal_mask(
    W,
    H,
    comp_used,
    sample_rate,
    n_fft,
    low_freq=300,
    high_freq=4000,
    threshold=0.45,
    eps=1e-10
):
    """
    Generate a frame-wise vocal/harmonic mask from NMF W and H.

    Parameters
    ----------
    W : ndarray, shape (K, N)
        NMF basis matrix.

    H : ndarray, shape (N, T)
        NMF activation matrix.

    sample_rate : int
        Audio sample rate.

    n_fft : int
        FFT size.

    threshold : float
        Minimum vocal score required for a component/frame to
        contribute to the vocal mask.

    Returns
    -------
    mask : ndarray, shape (K, T)
        Vocal mask in the range [0, 1].
    """

    W = np.maximum(W, 0)
    H = np.maximum(H, 0)

    K, num_components = W.shape
    T = H.shape[1]

    # Frequency axis
    freqs = np.linspace(
        0,
        sample_rate / 2,
        K
    )

    mid_mask = (
        (freqs >= low_freq) &
        (freqs <= high_freq)
    )

    # Final accumulated vocal reconstruction
    vocal_reconstruction = np.zeros((K, T))
    remaining_reconstruction = np.zeros((K, T))

    # Total NMF reconstruction
    total_reconstruction = W @ H

    for component_idx in range(num_components):
        if component_idx in comp_used:
            continue

        W_i = W[:, component_idx]
        H_i = H[component_idx]

        # ---------------------------------------------------------
        # Component spectrogram: frequency x time
        # ---------------------------------------------------------
        component = W_i[:, None] * H_i[None, :]

        # ---------------------------------------------------------
        # Frame-wise mid-band ratio
        # ---------------------------------------------------------
        total_energy = np.sum(component, axis=0) + eps
        mid_energy = np.sum(
            component[mid_mask, :],
            axis=0
        )

        mid_band_ratio = (
            mid_energy / total_energy
        )

        # ---------------------------------------------------------
        # Frame-wise formant-like strength
        #
        # Compare energy in overlapping vocal regions.
        # Strong concentration in these regions increases score.
        # ---------------------------------------------------------
        band1 = (freqs >= 300) & (freqs <= 1000)
        band2 = (freqs >= 800) & (freqs <= 2500)
        band3 = (freqs >= 1800) & (freqs <= 3500)

        e1 = np.sum(component[band1, :], axis=0)
        e2 = np.sum(component[band2, :], axis=0)
        e3 = np.sum(component[band3, :], axis=0)

        formant_energy = np.maximum(
            e1,
            np.maximum(e2, e3)
        )

        formant_strength = (
            formant_energy /
            total_energy
        )

        # ---------------------------------------------------------
        # Frame-wise spectral envelope smoothness
        # ---------------------------------------------------------
        normalized_spectrum = (
            component /
            (total_energy[None, :] + eps)
        )

        smoothness = 1.0 - (
            np.mean(
                np.abs(
                    np.diff(
                        normalized_spectrum,
                        axis=0
                    )
                ),
                axis=0
            )
        )

        smoothness = np.clip(
            smoothness,
            0,
            1
        )

        # ---------------------------------------------------------
        # Vocal score for EVERY frame
        # ---------------------------------------------------------
        vocal_score = (
            0.45 * mid_band_ratio +
            0.35 * formant_strength +
            0.20 * smoothness
        )

        # ---------------------------------------------------------
        # Evaluate score
        #
        # Only keep frames whose score is sufficiently vocal-like.
        # Smooth transition instead of hard binary cutoff.
        # ---------------------------------------------------------
        score = np.clip(
            (vocal_score - threshold) /
            (1.0 - threshold + eps),
            0,
            1
        )

        # Weight the actual component by the frame-wise score
        vocal_reconstruction += (
            component *
            score[None, :]
        )

        remaining_reconstruction += (
            component *
            (1.0 - score[None, :])
        )

    # -------------------------------------------------------------
    # Convert accumulated vocal reconstruction into a mask
    # -------------------------------------------------------------
    mask = (
        vocal_reconstruction**2 /
        (total_reconstruction + eps)**2
    )
    mask = np.clip(mask, 0, 1)
    vocal_mask = mask.T

    mask_r = (
        remaining_reconstruction**2 /
        (total_reconstruction + eps)**2
    )
    mask_r = np.clip(mask_r, 0, 1)
    non_vocal_mask = mask_r.T

    # Print info
    print("mask min:", np.min(mask))
    print("mask max:", np.max(mask))
    print("mask mean:", np.mean(mask))
    print("mask median:", np.median(mask))
    print("fraction > 0.5:", np.mean(mask > 0.5))

    return vocal_mask, non_vocal_mask


def evaluate_decomposition(X, W, H, verbose=False, epsilon=1e-10):
    """
    Evaluate an NMF decomposition X ≈ W @ H.

    Parameters
    ----------
    X : np.ndarray
        Original magnitude spectrogram, shape (K, T).

    W : np.ndarray
        NMF basis matrix, shape (K, R).

    H : np.ndarray
        NMF activation matrix, shape (R, T).

    verbose : bool
        If True, print the individual scores.

    epsilon : float
        Small value to avoid division by zero.

    Returns
    -------
    normalized_score : float
        Overall score in [0, 1].
        Higher is better.
    """

    X = np.maximum(X, 0)
    W = np.maximum(W, 0)
    H = np.maximum(H, 0)

    R = W.shape[1]

    # ---------------------------------------------------------
    # 1. Reconstruction quality
    # ---------------------------------------------------------

    reconstruction = W @ H

    reconstruction_error = (
        np.linalg.norm(X - reconstruction, 'fro')
        / (np.linalg.norm(X, 'fro') + epsilon)
    )

    # Convert error to a score.
    reconstruction_score = 1.0 / (1.0 + reconstruction_error)


    # ---------------------------------------------------------
    # 2. Component distinctness
    # ---------------------------------------------------------

    # Each component's actual contribution:
    #
    # component_r = W[:, r] outer H[r, :]
    #
    # Shape: (K, T)

    components = []

    for r in range(R):
        component = np.outer(W[:, r], H[r, :])
        component = component / (
            np.linalg.norm(component) + epsilon
        )
        components.append(component)

    components = np.asarray(components)

    similarities = []

    for i in range(R):
        for j in range(i + 1, R):

            similarity = np.sum(
                components[i] * components[j]
            )

            similarities.append(similarity)

    if similarities:
        mean_similarity = np.mean(similarities)
    else:
        mean_similarity = 0.0

    # Low similarity is good.
    distinctness_score = 1.0 - mean_similarity
    distinctness_score = np.clip(
        distinctness_score, 0.0, 1.0
    )


    # ---------------------------------------------------------
    # 3. Component activation diversity
    # ---------------------------------------------------------

    # Normalize each H row so its magnitude doesn't dominate
    # the comparison.

    H_norm = H / (
        np.linalg.norm(H, axis=1, keepdims=True) + epsilon
    )

    activation_similarities = []

    for i in range(R):
        for j in range(i + 1, R):

            similarity = np.dot(
                H_norm[i],
                H_norm[j]
            )

            activation_similarities.append(similarity)

    if activation_similarities:
        mean_activation_similarity = np.mean(
            activation_similarities
        )
    else:
        mean_activation_similarity = 0.0

    activation_diversity_score = (
        1.0 - mean_activation_similarity
    )

    activation_diversity_score = np.clip(
        activation_diversity_score,
        0.0,
        1.0
    )


    # ---------------------------------------------------------
    # 4. Spectral diversity
    # ---------------------------------------------------------

    W_norm = W / (
        np.linalg.norm(W, axis=0, keepdims=True) + epsilon
    )

    spectral_similarities = []

    for i in range(R):
        for j in range(i + 1, R):

            similarity = np.dot(
                W_norm[:, i],
                W_norm[:, j]
            )

            spectral_similarities.append(similarity)

    if spectral_similarities:
        mean_spectral_similarity = np.mean(
            spectral_similarities
        )
    else:
        mean_spectral_similarity = 0.0

    spectral_diversity_score = (
        1.0 - mean_spectral_similarity
    )

    spectral_diversity_score = np.clip(
        spectral_diversity_score,
        0.0,
        1.0
    )


    # ---------------------------------------------------------
    # Combine scores
    # ---------------------------------------------------------

    scores = {
        "reconstruction": reconstruction_score,
        "component_distinctness": distinctness_score,
        "activation_diversity": activation_diversity_score,
        "spectral_diversity": spectral_diversity_score,
    }

    # Equal weighting for now.
    normalized_score = np.mean(
        list(scores.values())
    )

    normalized_score = float(
        np.clip(normalized_score, 0.0, 1.0)
    )


    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    if verbose:

        print("\nDecomposition evaluation:")

        for name, score in scores.items():
            print(f"  {name:25s}: {score:.4f}")

        print(
            f"  {'NORMALIZED SCORE':25s}: "
            f"{normalized_score:.4f}"
        )

    return normalized_score