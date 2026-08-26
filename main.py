import os

import stft
import utils
import drum_mask

def main():
    # -------------------------
    # Configuration
    # -------------------------
    if not os.path.exists("Spectograms"):
        os.makedirs("Spectograms")

    input_file = "Audio/trimmed.wav"
    output_file = "Audio/reconstructed.wav"
    spectrogram_file = "Spectograms/spectrogram.png"
    duration = 10 # seconds

    frame_size = 1024
    hop_size = 512

    # -------------------------
    # Forward process
    # -------------------------

    sample_rate, audio = utils.load_audio(
        input_file,
        duration=duration
    )

    window = stft.create_window(frame_size)

    frames = stft.get_frames(
        audio,
        frame_size,
        hop_size
    )

    windowed_frames = stft.apply_window(
        frames,
        window
    )

    spectra = stft.calculate_fft(
        windowed_frames
    )

    # print(spectra.shape)

    magnitude = utils.calculate_magnitude(
        spectra
    )

    # Save spectrogram
    utils.save_spectrogram(
        magnitude,
        sample_rate,
        hop_size,
        spectrogram_file
    )

    # -------------------------
    # Apply mask
    # -------------------------

    mask = drum_mask.create_drum_mask_frequency_soft(
        spectra,
        threshold=2.5,
        full_strength=4
    )

    mask = drum_mask.smooth_mask_frequency(
        mask,
        kernel_size=10
    )

    # mask = drum_mask.create_drum_mask_simple(
    #     spectra,
    #     threshold=3.0,
    #     min_high_freq_ratio=0.25,
    #     neighbour_radius=2
    # )

    utils.save_mask(
        mask,
        sample_rate,
        hop_size,
        "Spectograms/drum_mask.png"
    )

    masked_spectra = utils.apply_mask(
        spectra,
        mask
    )

    masked_magnitude = utils.calculate_magnitude(
        masked_spectra
    )

    # Save spectrogram
    utils.save_spectrogram(
        masked_magnitude,
        sample_rate,
        hop_size,
        "Spectograms/masked_spectrogram.png"
    )

    # -------------------------
    # Inverse process
    # -------------------------

    reconstructed_frames = stft.calculate_ifft(
        masked_spectra,
        frame_size
    )

    reconstructed_audio = stft.overlap_add(
        reconstructed_frames,
        window,
        hop_size
    )

    # -------------------------
    # Save reconstructed audio
    # -------------------------

    utils.save_audio(
        output_file,
        sample_rate,
        reconstructed_audio
    )

    print("Done!")
    print("Original:", input_file)
    print("Reconstructed:", output_file)
    print("Spectrogram:", spectrogram_file)


if __name__ == "__main__":
    main()