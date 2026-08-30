import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# ============================================================
# 1. Frequency Prior
# ============================================================

def vocal_frequency_prior(freqs, low=80.0, high=8000.0):
    """
    Create a soft frequency prior for vocals.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency corresponding to each FFT bin.

    low : float
        Lower useful vocal frequency.

    high : float
        Upper useful vocal frequency.

    Returns
    -------
    prior : np.ndarray
        Shape: (frequency_bins,)
        Values between 0 and 1.
    """

    prior = np.zeros_like(freqs, dtype=float)

    # Fully active region
    inside = (freqs >= low) & (freqs <= high)
    prior[inside] = 1.0

    # Smooth fade-in below low
    fade_low = (freqs >= low * 0.5) & (freqs < low)

    if np.any(fade_low):
        prior[fade_low] = (
            (freqs[fade_low] - low * 0.5)
            / (low * 0.5)
        )

    # Smooth fade-out above high
    fade_high = (freqs > high) & (freqs <= high * 1.25)

    if np.any(fade_high):
        prior[fade_high] = (
            1.0
            - (freqs[fade_high] - high)
            / (high * 0.25)
        )

    return np.clip(prior, 0.0, 1.0)


# ============================================================
# 2. Estimate F0 using Harmonic Product
# ============================================================

def estimate_f0(
    spectrum,
    freqs,
    fmin=80.0,
    fmax=1000.0,
    max_harmonics=8
):
    """
    Estimate fundamental frequency using harmonic summation.

    Parameters
    ----------
    spectrum : np.ndarray
        Magnitude spectrum for one frame.

    freqs : np.ndarray
        FFT frequencies.

    fmin : float
        Minimum allowed F0.

    fmax : float
        Maximum allowed F0.

    max_harmonics : int
        Number of harmonics to evaluate.

    Returns
    -------
    f0 : float
        Estimated fundamental frequency.
        Returns 0 if no reliable F0 is found.
    """

    valid = (freqs >= fmin) & (freqs <= fmax)

    candidate_freqs = freqs[valid]

    if len(candidate_freqs) == 0:
        return 0.0

    scores = np.zeros(len(candidate_freqs))

    for i, f0 in enumerate(candidate_freqs):

        score = 0.0

        for harmonic in range(1, max_harmonics + 1):

            target = f0 * harmonic

            if target > freqs[-1]:
                break

            idx = np.argmin(np.abs(freqs - target))

            # Higher harmonics get slightly less weight
            weight = 1.0 / np.sqrt(harmonic)

            score += spectrum[idx] * weight

        scores[i] = score

    best_idx = np.argmax(scores)

    return candidate_freqs[best_idx]


# ============================================================
# 3. Harmonic Score
# ============================================================

def harmonic_score(
    spectrum,
    freqs,
    f0,
    max_harmonics=12,
    bandwidth=2
):
    """
    Calculate how strongly a spectrum follows the harmonics
    of a given F0.

    Returns
    -------
    score : np.ndarray
        Per-frequency harmonic confidence.
    """

    score = np.zeros_like(spectrum, dtype=float)

    if f0 <= 0:
        return score

    for harmonic in range(1, max_harmonics + 1):

        target = f0 * harmonic

        if target > freqs[-1]:
            break

        idx = np.argmin(np.abs(freqs - target))

        left = max(0, idx - bandwidth)
        right = min(len(freqs), idx + bandwidth + 1)

        # Energy around harmonic
        local_energy = np.max(spectrum[left:right])

        # Harmonic weight
        weight = 1.0 / np.sqrt(harmonic)

        score[left:right] += local_energy * weight

    return score


# ============================================================
# 4. Normalize Array
# ============================================================

def normalize(x):
    """
    Normalize an array to [0, 1].
    """

    maximum = np.max(x)

    if maximum <= 1e-12:
        return np.zeros_like(x)

    return x / maximum


# ============================================================
# 5. Build Vocal Confidence
# ============================================================

def build_vocal_confidence(
    magnitude,
    sample_rate,
    n_fft,
    fmin=80.0,
    fmax=1000.0,
    max_harmonics=12
):
    """
    Build a vocal confidence matrix.

    Parameters
    ----------
    magnitude : np.ndarray
        STFT magnitude.

        Shape:
            (frames, frequency_bins)

    sample_rate : int
        Audio sample rate.

    n_fft : int
        FFT size.

    Returns
    -------
    confidence : np.ndarray
        Vocal confidence matrix.

        Shape:
            (frames, frequency_bins)
    """

    frames, frequency_bins = magnitude.shape

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

    # Make sure dimensions agree
    freqs = freqs[:frequency_bins]

    # -----------------------------------------
    # Frequency prior
    # -----------------------------------------

    freq_prior = vocal_frequency_prior(
        freqs,
        low=80,
        high=8000
    )

    confidence = np.zeros_like(magnitude, dtype=float)

    f0_track = np.zeros(frames)

    # -----------------------------------------
    # Process every frame
    # -----------------------------------------

    for t in range(frames):

        spectrum = magnitude[t]

        # Normalize frame
        spectrum_norm = normalize(spectrum)

        # -------------------------------------
        # Estimate F0
        # -------------------------------------

        f0 = estimate_f0(
            spectrum_norm,
            freqs,
            fmin=fmin,
            fmax=fmax,
            max_harmonics=8
        )

        f0_track[t] = f0

        if f0 <= 0:
            continue

        # -------------------------------------
        # Harmonic structure
        # -------------------------------------

        h_score = harmonic_score(
            spectrum_norm,
            freqs,
            f0,
            max_harmonics=max_harmonics
        )

        h_score = normalize(h_score)

        # -------------------------------------
        # Combine harmonicity + frequency prior
        # -------------------------------------

        frame_confidence = (
            0.75 * h_score +
            0.25 * freq_prior
        )

        # Also require actual energy
        frame_confidence *= spectrum_norm

        confidence[t] = frame_confidence

    return confidence, f0_track


# ============================================================
# 6. F0 Continuity
# ============================================================

def apply_f0_continuity(
    confidence,
    f0_track,
    strength=0.7
):
    """
    Increase vocal confidence when F0 changes smoothly
    between neighboring frames.
    """

    frames, bins = confidence.shape

    continuity = np.zeros(frames)

    for t in range(frames):

        f0 = f0_track[t]

        if f0 <= 0:
            continue

        neighbors = []

        if t > 0 and f0_track[t - 1] > 0:
            neighbors.append(f0_track[t - 1])

        if t < frames - 1 and f0_track[t + 1] > 0:
            neighbors.append(f0_track[t + 1])

        if not neighbors:
            continue

        # Relative pitch difference
        differences = [
            abs(f0 - n) / f0
            for n in neighbors
        ]

        difference = np.mean(differences)

        # Small difference = strong continuity
        continuity[t] = np.exp(
            -difference * 30.0
        )

    continuity = continuity[:, np.newaxis]

    confidence = confidence * (
        (1.0 - strength) +
        strength * continuity
    )

    return confidence


# ============================================================
# 7. Smooth Mask
# ============================================================

def smooth_vocal_mask(
    mask,
    time_sigma=1.5,
    frequency_sigma=1.0
):
    """
    Smooth the vocal mask over time and frequency.
    """

    # Smooth across time
    mask = gaussian_filter1d(
        mask,
        sigma=time_sigma,
        axis=0
    )

    # Smooth across frequency
    mask = gaussian_filter1d(
        mask,
        sigma=frequency_sigma,
        axis=1
    )

    return np.clip(mask, 0.0, 1.0)


# ============================================================
# 8. Convert Confidence -> Mask
# ============================================================

def confidence_to_mask(
    confidence,
    strength=1.5
):
    """
    Convert vocal confidence into a soft mask.

    Higher strength makes the mask more selective.
    """

    confidence = np.maximum(confidence, 0.0)

    mask = confidence ** strength

    maximum = np.max(mask)

    if maximum > 0:
        mask /= maximum

    return np.clip(mask, 0.0, 1.0)


# ============================================================
# 9. MAIN FUNCTION
# ============================================================

def make_vocal_mask(
    magnitude,
    sample_rate,
    n_fft
):
    """
    Complete vocal-mask pipeline.

    Returns
    -------
    vocal_mask : np.ndarray
        Shape: (frames, frequency_bins)

    f0_track : np.ndarray
        Estimated vocal F0 for each frame.
    """

    # -----------------------------------------
    # Step 1: Build vocal confidence
    # -----------------------------------------

    confidence, f0_track = build_vocal_confidence(
        magnitude,
        sample_rate,
        n_fft
    )

    # -----------------------------------------
    # Step 2: Apply pitch continuity
    # -----------------------------------------

    confidence = apply_f0_continuity(
        confidence,
        f0_track,
        strength=0.7
    )

    # -----------------------------------------
    # Step 3: Convert confidence to mask
    # -----------------------------------------

    mask = confidence_to_mask(
        confidence,
        strength=0.7
    )

    # -----------------------------------------
    # Step 4: Smooth
    # -----------------------------------------

    mask = smooth_vocal_mask(
        mask,
        time_sigma=1.5,
        frequency_sigma=3.0
    )

    return mask, f0_track