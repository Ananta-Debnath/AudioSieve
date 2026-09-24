import os
import numpy as np

import utils
import stft
import nmf
import semi_supervised_nmf
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

    # Dictionary for separated spectra only
    spectra_dict = {}
    mag_dict = {}
    comp_dict = {}
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

    B, G = semi_supervised_nmf.separate_sources_nmf(
        X=magnitude.T,
        num_components=20,
    )

    print(f"B.shape: {B.shape}")
    print(f"G.shape: {G.shape}")

    semi_supervised_nmf.save_component_spectrograms(
        B,
        G,
        output_dir="Spectograms/nmf/new"
    )

    nmf_mag = sum(
        np.outer(B[:, k], G[k, :])
        for k in range(B.shape[1])
    )
    nmf_mag = nmf_mag.T

    # nmf.save_nmf_visualizations(
    #     B,
    #     G,
    #     sample_rate=44100,
    #     n_fft=1024,
    #     output_dir="Spectograms/nmf"
    # )

    df = nmf.analyze_nmf_components(
        B,
        G,
        sample_rate=sample_rate,
        n_fft=frame_size,
    )

    df_vocal = semi_supervised_nmf.analyze_vocal_features(
        B, G,
        sample_rate=sample_rate,
        n_fft=frame_size,
    )

    df = df.merge(
        df_vocal,
        on="component",
        how="left"
    )

    df = semi_supervised_nmf.score_components(df)

    # # nmf.rank_nmf_components(df)

    # nmf.save_nmf_analysis(df)

    # -------------------------
    # Separation
    # -------------------------
    PERCUSSION_THRESHOLD = 0.25
    BASS_THRESHOLD = 0.6
    VOCAL_THRESHOLD = 0.6
    # VOCAL_THRESHOLD = 0.45
    HARMONIC_THRESHOLD = 0.525

    # percussion
    # comp_dict["percussion"] = df.sort_values(by="percussion_score", ascending=False).head(2).index.tolist()
    comp_dict["percussion"] = df[df["percussion_score"] > PERCUSSION_THRESHOLD].index.tolist()
    print(f"percussion_comp: {comp_dict['percussion']}")

    # bass
    # comp_dict["bass"] = df.sort_values(by="bass_score", ascending=False).head(3).index.tolist()
    comp_dict["bass"] = df[df["bass_score"] > BASS_THRESHOLD].index.tolist()
    print(f"bass_comp: {comp_dict['bass']}")

    # vocal
    # comp_dict["vocal"] = df.sort_values(by="vocal_score", ascending=False).head(4).index.tolist()
    comp_dict["vocal"] = df[df["vocal_score"] > VOCAL_THRESHOLD].index.tolist()
    print(f"vocal_comp: {comp_dict['vocal']}")

    # harmonic
    # comp_dict["harmonic"] = df.sort_values(by="harmonic_score", ascending=False).head(6).index.tolist()
    # comp_dict["harmonic"] = df[df["harmonic_score"] > HARMONIC_THRESHOLD].index.tolist()
    # print(f"harmonic_comp: {comp_dict['harmonic']}")

    comp_used = sum(comp_dict.values(), [])
    comp_dict["remaining"] = [i for i in range(B.shape[1]) if i not in comp_used]
    print(f"remaining_comp: {comp_dict['remaining']}")

    for key, comp in comp_dict.items():
        mask_dict[key] = nmf_sep.get_custom_mask(
            B, G, nmf_mag,
            comps=comp,
        )

    # # keep only the unused ones
    # vocal_mask, non_vocal_mask = semi_supervised_nmf.get_framewise_vocal_mask(
    #     B, G, comp_used,
    #     sample_rate=sample_rate,
    #     n_fft=frame_size,
    #     threshold=VOCAL_THRESHOLD
    # )
    # mask_dict["vocal"] = vocal_mask
    # mask_dict["remaining"] = non_vocal_mask

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
    
    masks = nmf.nmf_component_masks(B, G)

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