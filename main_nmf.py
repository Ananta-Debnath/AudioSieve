import os
import numpy as np

import utils
import stft
import nmf
import nmf_sep

def main():
    # -------------------------
    # Configuration
    # -------------------------
    def make_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    make_dir("Spectograms")
    make_dir("Audio")
    make_dir("Spectograms/nmf")
    make_dir("Audio/nmf")
    make_dir("NMF")
    make_dir("Audio/NMF/groups")

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

    # Dictionary for separated spectra only
    spectra_dict = {}
    mag_dict = {}
    mask_dict = {}
    audio_dict = {}

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

    print(f"spectra.shape: {spectra.shape}")

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

    W, H, nmf_mag = nmf.calculate_nmf(
        spectra,
        n_components=30
    )

    print(f"W.shape: {W.shape}")
    print(f"H.shape: {H.shape}")

    nmf.save_nmf_visualizations(
        W,
        H,
        sample_rate=44100,
        n_fft=1024,
        output_dir="Spectograms/nmf"
    )

    df = nmf.analyze_nmf_components(
        W,
        H,
        sample_rate=44100,
        n_fft=1024
    )

    # nmf.rank_nmf_components(df)

    # nmf.save_nmf_analysis(df)

    # -------------------------
    # Separation
    # -------------------------

    # Bass
    df = nmf_sep.rank_bass_components(df)

    mask_dict["bass"] = nmf_sep.get_bass_mask(
        df, W, H, 
        spectra, 
        nmf_mag, 
        # top_n=3,
        # print_info=True
    )

    # Drum
    df = nmf_sep.calculate_drum_score(df)

    mask_dict["drum"] = nmf_sep.get_drum_mask(
        df, W, H,
        spectra,
        nmf_mag,
        # top_n=7,
        # print_info=True
    )

    mask_dict["rest"] = nmf_sep.get_rest_mask(mask_dict)

    for key, mask in mask_dict.items():
        spectra_dict[key] = utils.apply_mask(spectra, mask)

    for key, spec in spectra_dict.items():
        mag_dict[key] = utils.calculate_magnitude(spec)

    for key, mag in mag_dict.items():
        utils.save_spectrogram(
            mag,
            sample_rate,
            hop_size,
            spectogram_filename(key)
        )


    # -------------------------
    # Inverse process
    # -------------------------

    for key, spec in spectra_dict.items():
        audio_dict[key] = stft.calculate_istft(
            spec,
            frame_size,
            hop_size
        )

    # -------------------------
    # Save reconstructed audio
    # -------------------------

    for key, audio in audio_dict.items():
        utils.save_audio(
            audio_filename(key),
            sample_rate,
            audio
        )

    print("Done!")
    print("Original:", input_file)
    print("Reconstructed:", output_file)
    print("Spectrogram:", spectogram_filename())

    nmf.save_nmf_analysis(df)
    
    masks = nmf.nmf_component_masks(W, H)

    for mask_index, mask in enumerate(masks):

        spectra_dict[f"component_{mask_index}"] = spectra * mask

        audio = stft.calculate_istft(spectra_dict[f"component_{mask_index}"])

        audio *= 5

        utils.save_audio(
            audio_filename(f"nmf/component_{mask_index}"),
            sample_rate,
            audio
        )

if __name__ == "__main__":
    main()