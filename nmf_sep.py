import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rank_bass_components(df):
    """
    Rank NMF components by how bass-like they are.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing component-level NMF features.

    Returns
    -------
    pandas.DataFrame
        Copy of df with a 'bass_score' column, sorted from
        most bass-like to least bass-like.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # Normalize helper
    # ---------------------------------------------------------
    def normalize(series):
        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:
            return pd.Series(0.5, index=series.index)

        return (series - minimum) / (maximum - minimum)

    # ---------------------------------------------------------
    # Normalize relevant features
    # ---------------------------------------------------------
    bass_ratio = normalize(df["bass_ratio"])
    low_ratio = normalize(df["low_ratio"])
    centroid = normalize(df["spectral_centroid"])
    harmonicity = normalize(df["harmonicity"])
    sustain = normalize(df["sustain"])
    transientness = normalize(df["transientness"])

    # ---------------------------------------------------------
    # Bass score
    #
    # Higher:
    #   bass_ratio
    #   low_ratio
    #   harmonicity
    #   sustain
    #
    # Lower:
    #   spectral centroid
    #   transientness
    # ---------------------------------------------------------
    df["bass_score"] = (
        0.30 * bass_ratio +
        0.20 * low_ratio +
        0.20 * harmonicity +
        0.15 * sustain +
        0.10 * (1 - centroid) +
        0.05 * (1 - transientness)
    )

    # Highest score first
    # df = df.sort_values(
    #     "bass_score",
    #     ascending=False
    # ).reset_index(drop=True)

    return df


def get_bass_spectogram(df, W, H, spectra, nmf_mag, top_n=3, print_info=False):
    df = df.copy()
    df = df.sort_values(
        "bass_score",
        ascending=False
    ).reset_index(drop=True)

    top_bass = df.head(top_n)["component"].tolist()

    print(f"Top bass components: {top_bass}")
    
    bass_magnitude = sum(
        np.outer(W[:, k], H[k, :])
        for k in top_bass
    )

    bass_magnitude = bass_magnitude.T

    eps = 1e-10

    bass_power = bass_magnitude ** 2
    total_power = np.abs(spectra) ** 2

    bass_mask = bass_power / (total_power + eps)

    bass_mask = np.clip(bass_mask, 0, 1)

    bass_spectra = spectra * bass_mask

    if print_info:
        magnitude = np.abs(spectra)
        print(f"bass_magnitude.max(): {bass_magnitude.max()}")
        print(f"magnitude.max(): {magnitude.max()}")
        print(f"bass_mag > mag (mean): {np.mean(bass_magnitude > magnitude)}")

    return bass_spectra


def calculate_drum_score(df):

    def rank_score(column):
        values = df[column].astype(float).to_numpy()

        ranks = np.argsort(np.argsort(values))

        return ranks / max(len(values) - 1, 1)

    # Rank-normalized features
    energy = rank_score("energy")
    low_ratio = rank_score("low_ratio")
    high_ratio = rank_score("high_ratio")
    centroid = rank_score("spectral_centroid")
    bandwidth = rank_score("spectral_bandwidth")
    peakiness = rank_score("peakiness")
    transientness = rank_score("transientness")
    sustain = rank_score("sustain")
    harmonicity = rank_score("harmonicity")
    rhythmicity = rank_score("rhythmicity")
    num_peaks = rank_score("num_peaks")

    # --------------------------------------------------
    # LOW-FREQUENCY PERCUSSION
    # --------------------------------------------------

    low_percussion = (
        0.30 * low_ratio +
        0.20 * peakiness +
        0.20 * rhythmicity +
        0.15 * (1.0 - sustain) +
        0.10 * (1.0 - harmonicity) +
        0.05 * energy
    )

    # --------------------------------------------------
    # HIGH / METALLIC PERCUSSION
    # --------------------------------------------------

    high_percussion = (
        0.30 * high_ratio +
        0.20 * bandwidth +
        0.15 * rhythmicity +
        0.15 * (1.0 - sustain) +
        0.10 * num_peaks +
        0.10 * (1.0 - harmonicity)
    )

    # --------------------------------------------------
    # FINAL DRUM SCORE
    # --------------------------------------------------

    drum_score = np.maximum(
        low_percussion,
        high_percussion
    )

    df["drum_score"] = drum_score

    return df


def get_drum_spectogram(df, W, H, spectra, nmf_mag, top_n=5, print_info=False):

    df = df.copy()
    df = df.sort_values(
        "drum_score",
        ascending=False
    ).reset_index(drop=True)

    top_percussive = df.head(top_n)["component"].tolist()

    print(f"Top percussive components: {top_percussive}")

    eps = 1e-10

    drum_power = np.zeros_like(
        nmf_mag,
        dtype=float
    )

    other_power = np.zeros_like(
        nmf_mag,
        dtype=float
    )

    # selected = set(top_percussive)

    top_percussive = [28, 0]

    for k in range(W.shape[1]):

        component = np.outer(
            W[:, k],
            H[k, :]
        ).T

        component_power = component ** 2

        if k in top_percussive:
            drum_power += component_power
        else:
            other_power += component_power

    # Wiener-style soft mask
    drum_mask = drum_power / (
        drum_power +
        other_power +
        eps
    )

    drum_spectra = spectra * drum_mask

    if print_info:
        print("--------------------------DRUM---------------------------")
        print("Drum magnitude:")
        print("min:", drum_power.min())
        print("max:", drum_power.max())
        print("mean:", drum_power.mean())

        print("\nDrum mask:")
        print("min:", drum_mask.min())
        print("max:", drum_mask.max())
        print("mean:", drum_mask.mean())

        print("\nMask percentiles:")
        print(np.percentile(
            drum_mask,
            [50, 75, 90, 95, 99]
        ))
        print("---------------------------------------------------------")

    return drum_spectra