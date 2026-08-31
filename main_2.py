import os

import utils
import stft
import nmf
import drum_mask_gen
import vocal_mask_gen

def main():
    # -------------------------
    # Configuration
    # -------------------------
    if not os.path.exists("Spectograms"):
        os.makedirs("Spectograms")

    if not os.path.exists("Audio"):
        os.makedirs("Audio")

    if not os.path.exists("Spectograms/nmf"):
        os.makedirs("Spectograms/nmf")

    if not os.path.exists("Audio/nmf"):
        os.makedirs("Audio/nmf")

    input_file = "Audio/trimmed.wav"
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

    spectra = stft.calculatr_stft(
        audio,
        frame_size,
        hop_size
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
    # NMF
    # -------------------------

    W, H, reconstructed = nmf.calculate_nmf(
        spectra,
        n_components=30
    )

    reconstructed = nmf.reconstruct_spectra(
        W,
        H,
        spectra
    )

    print(W.shape)
    print(H.shape)
    print(reconstructed.shape)

    nmf.save_nmf_visualizations(
        W,
        H,
        sample_rate=44100,
        n_fft=1024,
        output_dir="Spectograms/nmf"
    )

    utils.save_spectrogram(
        utils.calculate_magnitude(reconstructed),
        sample_rate,
        hop_size,
        spectogram_filename("restored")
    )

    # -------------------------
    # Inverse process
    # -------------------------

    masks = nmf.nmf_component_masks(W, H)

    for mask_index, mask in enumerate(masks):

        component_spectra = spectra * mask

        audio = stft.calculate_istft(component_spectra)

        utils.save_audio(
            audio_filename(f"nmf/component_{mask_index}"),
            sample_rate,
            audio
        )

    reconstructed_audio = stft.calculate_istft(
        reconstructed,
        frame_size,
        hop_size
    )

    # -------------------------
    # Save reconstructed audio
    # -------------------------
    
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