import numpy as np

def create_frequency_mask(
    spectra,
    sample_rate,
    frame_size,
    cutoff_frequency
):
    """Create a simple low-pass binary frequency mask."""

    frequencies = np.fft.rfftfreq(
        frame_size,
        d=1 / sample_rate
    )

    mask = frequencies < cutoff_frequency

    # Expand mask to all time frames
    mask = np.broadcast_to(
        mask,
        spectra.shape
    )

    return mask


def create_drum_mask(
    spectra,
    threshold=2.0
):
    """
    Create a simple drum mask based on
    sudden increases in broadband energy.
    """

    # Magnitude of each frequency bin
    magnitude = np.abs(spectra)

    # Broadband energy for each time frame
    energy = np.mean(magnitude, axis=1)

    # Compare each frame with previous frame
    previous_energy = np.roll(energy, 1)

    # Avoid treating the first frame as a transient
    previous_energy[0] = energy[0]

    # Energy increase ratio
    increase_ratio = (
        energy / (previous_energy + 1e-10)
    )

    # Detect transient frames
    transient_frames = increase_ratio > threshold

    # Create mask with same shape as spectra
    mask = np.zeros_like(magnitude)

    # Keep all frequencies during transient frames
    mask[transient_frames, :] = 1.0

    return mask


def create_drum_mask_median(
    spectra,
    window_frames=15,
    threshold=2.0
):
    """
    Create a simple drum mask using broadband
    energy compared against a local median.
    """

    # Magnitude of each frequency bin
    magnitude = np.abs(spectra)

    # Broadband energy for each time frame
    energy = np.mean(magnitude, axis=1)

    # Make sure the window size is odd
    if window_frames % 2 == 0:
        window_frames += 1

    half_window = window_frames // 2

    # Pad the energy at the boundaries
    padded_energy = np.pad(
        energy,
        (half_window, half_window),
        mode="edge"
    )

    # Local median
    local_median = np.zeros_like(energy)

    for i in range(len(energy)):
        start = i
        end = i + window_frames

        local_median[i] = np.median(
            padded_energy[start:end]
        )

    # Compare current energy against local baseline
    energy_ratio = (
        energy / (local_median + 1e-10)
    )

    # Detect transient frames
    transient_frames = energy_ratio > threshold

    # Create time-frequency mask
    mask = np.zeros_like(magnitude)

    mask[transient_frames, :] = 1.0

    return mask


def create_drum_mask_frequency(
    spectra,
    threshold=1.35
):
    """
    Create a frequency-dependent transient mask.

    Each frequency bin is compared against
    the same frequency bin in the previous frame.
    """

    magnitude = np.abs(spectra)

    # Previous frame
    previous_magnitude = np.roll(
        magnitude,
        1,
        axis=0
    )

    # Don't classify the first frame
    previous_magnitude[0] = magnitude[0]

    # Calculate increase independently
    # for every time-frequency bin
    ratio = (
        magnitude /
        (previous_magnitude + 1e-10)
    )

    # Create binary mask
    mask = (ratio > threshold).astype(float)

    return mask


def create_drum_mask_frequency_soft(spectra, threshold=1.35, full_strength=2.5):
    """
    Create a soft frequency-dependent transient mask.

    threshold:
        Ratio at which a frequency bin starts being retained.

    full_strength:
        Ratio at which the mask reaches 1.0.
    """

    magnitude = np.abs(spectra)

    # Previous frame
    previous_magnitude = np.roll(
        magnitude,
        1,
        axis=0
    )

    # Don't classify the first frame
    previous_magnitude[0] = magnitude[0]

    # Frequency-dependent increase ratio
    ratio = (
        magnitude /
        (previous_magnitude + 1e-10)
    )

    # Convert ratio into a soft 0-1 mask
    mask = (
        (ratio - threshold) /
        (full_strength - threshold)
    )

    # Limit to [0, 1]
    mask = np.clip(mask, 0.0, 1.0)

    return mask


def smooth_mask_frequency(mask, kernel_size=7):
    """
    Smooth a time-frequency mask across frequency.

    This encourages neighboring frequency bins
    to behave similarly.
    """

    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2

    padded = np.pad(
        mask,
        (
            (0, 0),
            (pad, pad)
        ),
        mode="edge"
    )

    smoothed = np.zeros_like(mask)

    for f in range(mask.shape[1]):
        start = f
        end = f + kernel_size

        smoothed[:, f] = np.mean(
            padded[:, start:end],
            axis=1
        )

    return smoothed


import numpy as np


def find_drum_stripes(
    spectra,
    threshold=3.0,
    min_high_freq_ratio=0.25,
    low_freq_bin=0,
    high_freq_bin=None,
):
    """
    Find candidate drum events from vertical spectral structures.

    Parameters
    ----------
    spectra : np.ndarray
        Magnitude STFT, shape (freq_bins, time_frames).

    threshold : float
        Number of local standard deviations above the frequency-bin
        baseline required for a bin to be considered strong.

    min_high_freq_ratio : float
        Minimum fraction of the upper-frequency region that must contain
        significant energy for a time frame to be considered a candidate.

    low_freq_bin : int
        Lowest frequency bin to inspect.

    high_freq_bin : int or None
        Highest frequency bin to inspect.

    Returns
    -------
    candidate : np.ndarray
        Boolean array of shape (time_frames,).
        True indicates a candidate vertical drum stripe.
    """

    if high_freq_bin is None:
        high_freq_bin = spectra.shape[0]

    region = spectra[low_freq_bin:high_freq_bin]

    # Robust-ish frequency baseline.
    baseline = np.median(region, axis=1, keepdims=True)
    spread = np.std(region, axis=1, keepdims=True) + 1e-8

    # Which frequency bins are unusually strong?
    strong = region > baseline + threshold * spread

    # We care about energy reaching upward through the spectrum.
    high_start = region.shape[0] // 2
    high_region = strong[high_start:]

    high_freq_ratio = np.mean(high_region, axis=0)

    candidate = high_freq_ratio >= min_high_freq_ratio

    return candidate


def create_drum_mask_simple(
    spectra,
    threshold=3.0,
    min_high_freq_ratio=0.25,
    neighbour_radius=2,
    low_freq_bin=0,
    high_freq_bin=None,
):
    """
    Create a soft drum mask using vertical spectral structures and
    neighbouring time frames.

    The mask is estimated independently for every frequency/time bin.
    """

    n_freq, n_time = spectra.shape
    magnitude = np.abs(spectra)

    if high_freq_bin is None:
        high_freq_bin = n_freq

    # ------------------------------------------------------------
    # 1. Find candidate vertical drum events
    # ------------------------------------------------------------

    candidate = find_drum_stripes(
        magnitude,
        threshold=threshold,
        min_high_freq_ratio=min_high_freq_ratio,
        low_freq_bin=low_freq_bin,
        high_freq_bin=high_freq_bin,
    )

    mask = np.zeros_like(magnitude, dtype=float)

    # ------------------------------------------------------------
    # 2. For every candidate event, compare it with neighbours
    # ------------------------------------------------------------

    for t in np.flatnonzero(candidate):

        left = max(0, t - neighbour_radius)
        right = min(n_time, t + neighbour_radius + 1)

        neighbours = np.concatenate([
            magnitude[left:t, :],
            magnitude[t + 1:right, :]
        ], axis=0)

        if neighbours.shape[1] == 0:
            continue

        # Estimate what the non-drum/background energy looks like
        # around this event.
        neighbour_level = np.median(neighbours, axis=0)

        current = magnitude[t, :]

        # Energy above the neighbouring level.
        excess = current - neighbour_level

        # Convert excess into a soft 0..1 drum probability.
        strength = np.maximum(excess, 0.0)

        mask[t, :] = strength / (current + 1e-8)

    return mask