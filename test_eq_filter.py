"""Sanity check for effects/eq_filter.py without the web UI.

Runs filter and EQ modes on a test clip and saves:
- gain curves (gain vs. frequency)       -> Spectograms/eq_curve_*.png
- before/after spectrograms              -> Spectograms/eq_*_spectrogram.png
- processed audio for listening          -> Audio/eq_*.wav
and prints band-energy numbers so the effect can be checked numerically.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import stft
import utils
from effects import eq_filter

INPUT_FILE = "Audio/sample-15s.wav"
DURATION = 10  # seconds

FRAME_SIZE = eq_filter.FRAME_SIZE
HOP_SIZE = eq_filter.HOP_SIZE

CASES = {
    "lowpass_500": {"mode": "filter", "filter_type": "lowpass", "cutoff": 500, "order": 4},
    "highpass_500": {"mode": "filter", "filter_type": "highpass", "cutoff": 500, "order": 4},
    "bandpass_300_3000": {"mode": "filter", "filter_type": "bandpass",
                          "low_cutoff": 300, "high_cutoff": 3000, "order": 4},
    "eq_bass_boost": {"mode": "eq", "bands": [
        {**band, "gain_db": 12 if band["name"] == "Bass" else 0}
        for band in eq_filter.DEFAULT_BANDS
    ]},
    "eq_mixed": {"mode": "eq", "bands": [
        {**band, "gain_db": g}
        for band, g in zip(eq_filter.DEFAULT_BANDS, [6, -6, 0, 4, -12])
    ]},
    "bandstop_800_1200": {"mode": "filter", "filter_type": "bandstop",
                          "low_cutoff": 800, "high_cutoff": 1200, "order": 2},
    "both_bright_no_rumble": {"mode": "both",
                              "bands": eq_filter.PRESETS["bright"]["bands"],
                              "filter_type": "highpass", "cutoff": 80, "order": 4},
    **{f"preset_{name}": {"preset": name} for name in eq_filter.PRESETS},
}


def save_curve(freqs, curve, title, filename):
    fig, (ax_lin, ax_db) = plt.subplots(1, 2, figsize=(12, 4))

    ax_lin.plot(freqs, curve)
    ax_lin.set_xscale("symlog", linthresh=100)
    ax_lin.set_xlabel("Frequency (Hz)")
    ax_lin.set_ylabel("Gain (linear)")
    ax_lin.grid(True, which="both", alpha=0.3)

    ax_db.plot(freqs, 20 * np.log10(curve + 1e-10))
    ax_db.set_xscale("symlog", linthresh=100)
    ax_db.set_ylim(-60, 15)
    ax_db.set_xlabel("Frequency (Hz)")
    ax_db.set_ylabel("Gain (dB)")
    ax_db.grid(True, which="both", alpha=0.3)

    fig.suptitle(title)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def band_energy_db(spectra, sample_rate, low, high):
    freqs = np.fft.rfftfreq(FRAME_SIZE, d=1 / sample_rate)
    band = (freqs >= low) & (freqs < high)
    energy = np.sum(np.abs(spectra[:, band]) ** 2)
    return 10 * np.log10(energy + 1e-20)


def main():
    os.makedirs("Spectograms", exist_ok=True)

    sample_rate, audio = utils.load_audio(INPUT_FILE, duration=DURATION)

    original_spectra = stft.calculatr_stft(audio, FRAME_SIZE, HOP_SIZE)
    utils.save_spectrogram(
        utils.calculate_magnitude(original_spectra),
        sample_rate,
        HOP_SIZE,
        "Spectograms/eq_original_spectrogram.png"
    )

    ranges = [(0, 250), (250, 2000), (2000, 8000), (8000, sample_rate / 2)]
    header = "".join(f"{f'{lo:g}-{hi:g} Hz':>16}" for lo, hi in ranges)
    print(f"{'case':<28}{header}{'peak':>8}")
    print(f"{'(dB change vs original)':<28}")

    for name, params in CASES.items():
        freqs, curve = eq_filter.build_curve(sample_rate, params, FRAME_SIZE)
        save_curve(freqs, curve, name, f"Spectograms/eq_curve_{name}.png")

        processed, _ = eq_filter.process(audio, sample_rate, params)
        utils.save_audio(f"Audio/eq_{name}.wav", sample_rate, processed)

        processed_spectra = stft.calculatr_stft(processed, FRAME_SIZE, HOP_SIZE)
        utils.save_spectrogram(
            utils.calculate_magnitude(processed_spectra),
            sample_rate,
            HOP_SIZE,
            f"Spectograms/eq_{name}_spectrogram.png"
        )

        changes = "".join(
            f"{band_energy_db(processed_spectra, sample_rate, lo, hi) - band_energy_db(original_spectra, sample_rate, lo, hi):>+16.1f}"
            for lo, hi in ranges
        )
        print(f"{name:<28}{changes}{np.max(np.abs(processed)):>8.3f}")

    print("\nSaved curves/spectrograms to Spectograms/ and audio to Audio/eq_*.wav")


if __name__ == "__main__":
    main()
