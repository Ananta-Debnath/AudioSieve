import numpy as np
import matplotlib.pyplot as plt
import os

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