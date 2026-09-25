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


def visualize_B_diff(B_init, B, output_dir="Spectrograms/nmf/B_diff"):
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


def separate_sources_nmf(X, num_components, alpha=100.0, beta=0.0, max_iter=300, tol=1e-4, diagnostics=True):
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
    diagnostics : bool - Save the B_diff plots and print the decomposition score
        (the web app turns this off; B and G are the same either way).
    
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

    if diagnostics:
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