import os

import stft
import utils
import drum_mask_gen
import vocal_mask_gen

def main():
    # -------------------------
    # Configuration
    # -------------------------
    if not os.path.exists("Spectograms"):
        os.makedirs("Spectograms")

    input_file = "Audio/trimmed.wav"
    drum_output_file = "Audio/drums.wav"
    output_file = "Audio/reconstructed.wav"
    duration = 10 # seconds

    frame_size = 1024
    hop_size = 512

    def spectogram_filename(name=None):
        if name:
            return f"Spectograms/{name}_spectrogram.png"
        else:
            return "Spectograms/spectrogram.png"

    def mask_filename(name=None):
        if name:
            return f"Spectograms/{name}_mask.png"
        else:
            return "Spectograms/mask.png"

    def audio_filename(name=None):
        if name:
            return f"Audio/{name}.wav"
        else:
            return "Audio/reconstructed.wav"

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

    print(spectra.shape)

    magnitude = utils.calculate_magnitude(
        spectra
    )

    # Save spectrogram
    utils.save_spectrogram(
        magnitude,
        sample_rate,
        hop_size,
        spectogram_filename("original")
    )

    # -------------------------
    # Apply mask
    # -------------------------

    # Drum
    drum_mask = drum_mask_gen.create_drum_mask_frequency_soft(
        spectra,
        threshold=2,
        full_strength=4
    )

    utils.save_mask(
        drum_mask,
        sample_rate,
        hop_size,
        mask_filename("drum")
    )

    drum_spectra = utils.apply_mask(
        spectra,
        drum_mask
    )

    reconstructed = utils.apply_mask(
        spectra,
        1-drum_mask
    )

    masked_magnitude = utils.calculate_magnitude(
        drum_spectra
    )

    rest_magnitude = utils.calculate_magnitude(
        reconstructed
    )

    # Vocal
    n_fft = (spectra.shape[1] - 1) * 2

    vocal_mask, f0_track = vocal_mask_gen.make_vocal_mask(
        rest_magnitude,
        sample_rate,
        n_fft
    )

    utils.save_mask(
        vocal_mask,
        sample_rate,
        hop_size,
        mask_filename("vocal")
    )

    vocal_spectra = utils.apply_mask(
        reconstructed,
        vocal_mask
    )

    vocal_magnitude = utils.calculate_magnitude(
        vocal_spectra
    )

    reconstructed = utils.apply_mask(
        reconstructed,
        1-vocal_mask
    )

    rest_magnitude = utils.calculate_magnitude(
        reconstructed
    )

    # Save spectrogram
    utils.save_spectrogram(
        masked_magnitude,
        sample_rate,
        hop_size,
        spectogram_filename("drum")
    )

    utils.save_spectrogram(
        vocal_magnitude,
        sample_rate,
        hop_size,
        spectogram_filename("vocal")
    )

    utils.save_spectrogram(
        rest_magnitude,
        sample_rate,
        hop_size,
        spectogram_filename("remaining")
    )

    # -------------------------
    # Inverse process
    # -------------------------

    drum_frames = stft.calculate_ifft(
        drum_spectra,
        frame_size
    )

    drum_audio = stft.overlap_add(
        drum_frames,
        window,
        hop_size
    )

    vocal_frames = stft.calculate_ifft(
        vocal_spectra,
        frame_size
    )

    vocal_audio = stft.overlap_add(
        vocal_frames,
        window,
        hop_size
    )

    reconstructed_frames = stft.calculate_ifft(
        reconstructed,
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
        audio_filename("drum"),
        sample_rate,
        drum_audio
    )

    utils.save_audio(
        audio_filename("vocal"),
        sample_rate,
        vocal_audio
    )
    

    utils.save_audio(
        audio_filename("remaining"),
        sample_rate,
        reconstructed_audio
    )

    print("Done!")
    print("Original:", input_file)
    print("Reconstructed:", output_file)
    print("Spectrogram:", spectogram_filename())


if __name__ == "__main__":
    main()