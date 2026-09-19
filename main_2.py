import os
import numpy as np

import utils
import stft
import nmf
import nmf_sep
import drum_mask_gen
import vocal_mask_gen

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

    reconstructed = nmf.reconstruct_spectra(
        W,
        H,
        spectra
    )

    print(f"W.shape: {W.shape}")
    print(f"H.shape: {H.shape}")
    print(f"reconstructed.shape: {reconstructed.shape}")
    print()

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

    df = nmf.analyze_nmf_components(
        W,
        H,
        sample_rate=44100,
        n_fft=1024
    )

    similarity_df = nmf.make_similarity_matrix(
        df,
        "NMF/component_similarity.csv"
    )

    # nmf.rank_nmf_components(df)

    # nmf.save_nmf_analysis(df)

    # -------------------------
    # Separation
    # -------------------------

    # Bass
    df = nmf_sep.rank_bass_components(df)

    bass_spectra = nmf_sep.get_bass_spectogram(
        df, W, H, 
        spectra, 
        nmf_mag, 
        # top_n=3,
        # print_info=True
    )

    bass_magnitude = utils.calculate_magnitude(bass_spectra)

    utils.save_spectrogram(
        bass_magnitude,
        sample_rate,
        hop_size,
        spectogram_filename("bass")
    )

    # Drum
    df = nmf_sep.calculate_drum_score(df)

    # print(
    #     df[
    #         [
    #             "component",
    #             "drum_score",
    #             "low_ratio",
    #             "high_ratio",
    #             "spectral_centroid",
    #             "spectral_bandwidth",
    #             "peakiness",
    #             "rhythmicity",
    #             "sustain",
    #             "harmonicity"
    #         ]
    #     ]
    #     .sort_values("drum_score", ascending=False)
    #     .to_string(index=False)
    # )

    drum_spectra = nmf_sep.get_drum_spectogram(
        df, W, H,
        spectra,
        nmf_mag,
        # top_n=7,
        # print_info=True
    )

    drum_magnitude = utils.calculate_magnitude(drum_spectra)

    utils.save_spectrogram(
        drum_magnitude,
        sample_rate,
        hop_size,
        spectogram_filename("drum")
    )


    # -------------------------
    # Inverse process
    # -------------------------

    nmf.save_nmf_analysis(df)

    masks = nmf.nmf_component_masks(W, H)

    for mask_index, mask in enumerate(masks):

        component_spectra = spectra * mask

        audio = stft.calculate_istft(component_spectra)

        audio *= 5

        utils.save_audio(
            audio_filename(f"nmf/component_{mask_index}"),
            sample_rate,
            audio
        )


    bass_audio = stft.calculate_istft(
        bass_spectra,
        frame_size,
        hop_size
    )

    drum_audio = stft.calculate_istft(
        drum_spectra,
        frame_size,
        hop_size
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
        audio_filename("bass"),
        sample_rate,
        bass_audio
    )

    utils.save_audio(
        audio_filename("drum"),
        sample_rate,
        drum_audio
    )
    
    utils.save_audio(
        audio_filename("restored"),
        sample_rate,
        reconstructed_audio
    )

    print("Done!")
    print("Original:", input_file)
    print("Reconstructed:", output_file)
    print("Spectrogram:", spectogram_filename())


    groups = [
        [0, 1, 2, 4, 8],
        [3, 7, 10, 13, 19],
        [5, 6, 17],
        [9, 12, 16, 18, 20, 21, 22, 27, 29],
        [11, 24],
        [14, 23, 25],
        [26, 28]
    ]

    for group_index, group in enumerate(groups):

        group_mask = np.maximum.reduce([
            masks[i]
            for i in group
        ])

        group_spectra = spectra * group_mask

        audio = stft.calculate_istft(group_spectra)

        utils.save_audio(
            audio_filename(f"NMF/groups/group_{group_index}"),
            sample_rate,
            audio
        )

    comps = [9, 10, 11, 21, 22]
    custom_spectra = nmf_sep.get_custom_spectra(
        W, H,
        spectra,
        comps
    )

    utils.save_spectrogram(
        utils.calculate_magnitude(custom_spectra),
        sample_rate,
        hop_size,
        spectogram_filename("custom")
    )

    custom_audio = stft.calculate_istft(
        custom_spectra,
        frame_size,
        hop_size
    )

    utils.save_audio(
        audio_filename("custom"),
        sample_rate,
        custom_audio
    )


if __name__ == "__main__":
    main()