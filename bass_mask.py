import numpy as np

from drum_mask import smooth_mask_frequency


def create_bass_mask_bandlimit(
    spectra,
    sample_rate,
    freq_low=40,
    freq_high=400,
    rolloff_bins=5
):
    """
    Create a soft band-pass mask isolating the bass fundamental range.

    Bass is a narrow low-frequency, sustained signal rather than a
    broadband transient, so unlike the drum masks this doesn't look at
    energy changes over time at all - it's a static frequency-shape
    mask. Bins inside [freq_low, freq_high] are weighted at (or near)
    1.0. Bins outside ramp down smoothly over `rolloff_bins` bins using
    a raised-cosine taper instead of a hard cutoff, since a brick-wall
    band-pass rings (introduces artifacts) at the edges.
    """

    n_freq_bins = spectra.shape[-1]
    frame_size = 2 * (n_freq_bins - 1)

    frequencies = np.fft.rfftfreq(
        frame_size,
        d=1 / sample_rate
    )

    # Bin index where the passband starts/ends
    low_bin = np.searchsorted(frequencies, freq_low)
    high_bin = np.searchsorted(frequencies, freq_high)

    # Distance (in bins) outside the passband, 0 if inside it
    bin_indices = np.arange(n_freq_bins)

    distance = np.zeros(n_freq_bins)
    distance = np.where(
        bin_indices < low_bin,
        low_bin - bin_indices,
        distance
    )
    distance = np.where(
        bin_indices >= high_bin,
        bin_indices - high_bin + 1,
        distance
    )

    # Raised-cosine rolloff: 1.0 inside the band, smoothly down to
    # 0.0 once `distance` reaches rolloff_bins.
    rolloff_bins = max(rolloff_bins, 1)
    taper = np.clip(distance, 0, rolloff_bins) / rolloff_bins

    mask_1d = 0.5 * (1 + np.cos(np.pi * taper))

    # Expand mask to all time frames
    mask = np.broadcast_to(
        mask_1d,
        spectra.shape
    )

    return mask


def create_bass_mask_energy(
    frame_spectra,
    sample_rate,
    freq_low=40,
    freq_high=250,
    low_threshold=0.12,
    high_threshold=0.30,
    rolloff_bins=5
):
    """
    Create a soft bass mask that also gates on how much of each frame's
    energy actually sits in the bass band.

    The static band-pass in create_bass_mask_bandlimit alone still
    passes a kick drum's fundamental, since it lives in the same
    40-400 Hz range as the bass. This adds a per-frame gate on top of
    that band shape: frames where the bass band only accounts for a
    small share of the frame's total energy (e.g. a drum-only
    breakdown) get suppressed, even though they fall inside the target
    frequency range.

    low_threshold / high_threshold:
        Fraction of a frame's total energy that must sit in the bass
        band before the gate opens. Below low_threshold the frame is
        fully suppressed; above high_threshold it passes through
        untouched; in between the gate ramps linearly.
    """

    magnitude = np.abs(frame_spectra)

    band_mask = create_bass_mask_bandlimit(
        frame_spectra,
        sample_rate,
        freq_low=freq_low,
        freq_high=freq_high,
        rolloff_bins=rolloff_bins
    )

    # How much of each frame's energy sits inside the bass band
    band_energy = np.sum(magnitude * band_mask, axis=1)
    total_energy = np.sum(magnitude, axis=1)

    relative_energy = band_energy / (total_energy + 1e-10)

    # Soft gate: 0 below low_threshold, 1 above high_threshold
    gate = (
        (relative_energy - low_threshold) /
        (high_threshold - low_threshold)
    )
    gate = np.clip(gate, 0.0, 1.0)

    # Apply the per-frame gate on top of the static band shape
    mask = band_mask * gate[:, np.newaxis]

    return mask
