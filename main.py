import os

import stft
import utils

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

    magnitude = stft.calculate_magnitude(
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
    # Inverse process
    # -------------------------

    reconstructed_frames = stft.calculate_ifft(
        spectra,
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