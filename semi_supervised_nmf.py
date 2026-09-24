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

    return pd.DataFrame(results)


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
        0.30 * sustain +
        0.25 * (1 - transientness) +
        0.20 * harmonicity +
        0.10 * (1 - bass_ratio) +
        0.10 * spectral_centroid +
        0.05 * spectral_bandwidth
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