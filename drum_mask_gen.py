import numpy as np
from scipy.ndimage import binary_opening


def clean_drum_mask(mask, threshold=0.15, time_size=3, freq_size=2):
    binary_mask = mask >= threshold

    structure = np.ones((time_size, freq_size))

    cleaned = binary_opening(
        binary_mask,
        structure=structure
    )

    return mask * cleaned


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
    # previous_magnitude = np.roll(
    #     magnitude,
    #     1,
    #     axis=0
    # )

    previous_magnitude = np.median(
        np.stack([
            np.roll(magnitude, 1, axis=0),
            np.roll(magnitude, 2, axis=0),
            np.roll(magnitude, 3, axis=0),
            np.roll(magnitude, 4, axis=0),
            np.roll(magnitude, 5, axis=0),
        ]),
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

    mask = clean_drum_mask(
        mask,
        threshold=0.1,
        time_size=1,
        freq_size=4
    )

    mask = smooth_mask_frequency(
        mask,
        kernel_size=5
    )

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