import os
import numpy as np

import utils
import stft
import semi_supervised_nmf
import nmf_sep

def main():
    print("Starting NMF-based source separation...")
    print()
    # -------------------------
    # Configuration
    # -------------------------
    def make_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    make_dir("Spectrograms")
    make_dir("Audio")
    make_dir("Spectrograms/nmf")
    make_dir("Audio/nmf")
    # make_dir("NMF")
    # make_dir("Audio/NMF/groups")

    input_file = "Audio/trimmed.wav"
    input_file = "Audio/audio.wav"
    output_file = "Audio/reconstructed.wav"
    # duration = 10 # seconds
    duration = None

    FRAME_TIME = 40 # ms
    OVERLAP = 0.5 # 50% overlap
    NUM_COMPONENTS = 20

    frame_size = 1024
    hop_size = 512

    def spectogram_filename(name=None):
        if name:
            return f"Spectrograms/{name}_spectrogram.png"
        else:
            return "Spectrograms/spectrogram.png"

    def mask_filename(name=None):
        if name:
            return f"Spectrograms/{name}_mask.png"
        else:
            return "Spectrograms/mask.png"

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

    def next_power_of_two(x):
        return 1 << (x - 1).bit_length()

    frame_size = int(sample_rate * FRAME_TIME / 1000)
    frame_size = next_power_of_two(frame_size)
    hop_size = int(frame_size * (1 - OVERLAP))
    print(f"frame_size: {frame_size}, hop_size: {hop_size}")
    print()

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
        num_components=NUM_COMPONENTS,
    )

    print(f"B.shape: {B.shape}")
    print(f"G.shape: {G.shape}")
    print()

    semi_supervised_nmf.save_component_spectrograms(
        B,
        G,
        output_dir="Spectrograms/nmf/new"
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
    #     output_dir="Spectrograms/nmf"
    # )

    df = nmf_sep.analyze_nmf_components(
        B,
        G,
        sample_rate=sample_rate,
        n_fft=frame_size,
    )

    df_vocal = nmf_sep.analyze_vocal_features(
        B, G,
        sample_rate=sample_rate,
        n_fft=frame_size,
    )

    df = df.merge(
        df_vocal,
        on="component",
        how="left"
    )

    df = nmf_sep.score_components(df)

    # # nmf.rank_nmf_components(df)

    # nmf.save_nmf_analysis(df)

    # -------------------------
    # Separation
    # -------------------------
    PERCUSSION_THRESHOLD = 0.25
    PERCUSSION_MIN = 2
    BASS_THRESHOLD = 0.6
    VOCAL_THRESHOLD = 0.62
    # VOCAL_THRESHOLD = 0.45
    HARMONIC_THRESHOLD = 0.525

    # percussion
    comp_dict["percussion"] = df.sort_values(by="percussion_score", ascending=False).head(PERCUSSION_MIN).index.tolist()
    # comp_dict["percussion"] = df[df["percussion_score"] > PERCUSSION_THRESHOLD].index.tolist()
    # if len(comp_dict["percussion"]) < PERCUSSION_MIN:
    #     comp_dict["percussion"] = df.sort_values(by="percussion_score", ascending=False).head(PERCUSSION_MIN).index.tolist()
    print(f"percussion_comp: {comp_dict['percussion']}")

    # bass
    # comp_dict["bass"] = df.sort_values(by="bass_score", ascending=False).head(3).index.tolist()
    comp_dict["bass"] = df[df["bass_score"] > BASS_THRESHOLD].index.tolist()
    print(f"bass_comp: {comp_dict['bass']}")

    comp_used = sum(comp_dict.values(), [])
    VOCAL_MAX = (NUM_COMPONENTS - len(comp_used)) * (3/5)
    VOCAL_MAX = int(VOCAL_MAX)

    # vocal
    comp_dict["vocal"] = df.sort_values(by="vocal_score", ascending=False).head(VOCAL_MAX).index.tolist()
    # comp_dict["vocal"] = df[df["vocal_score"] > VOCAL_THRESHOLD].index.tolist()
    # if len(comp_dict["vocal"]) > VOCAL_MAX:
    #     comp_dict["vocal"] = df.sort_values(by="vocal_score", ascending=False).head(VOCAL_MAX).index.tolist()
    print(f"vocal_comp: {comp_dict['vocal']}")

    # harmonic
    # comp_dict["harmonic"] = df.sort_values(by="harmonic_score", ascending=False).head(6).index.tolist()
    # comp_dict["harmonic"] = df[df["harmonic_score"] > HARMONIC_THRESHOLD].index.tolist()
    # print(f"harmonic_comp: {comp_dict['harmonic']}")

    comp_used = sum(comp_dict.values(), [])
    comp_dict["remaining"] = [i for i in range(B.shape[1]) if i not in comp_used]
    print(f"remaining_comp: {comp_dict['remaining']}")

    print()

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

    # mask_dict["rest"] = nmf_sep.get_rest_mask(mask_dict)
    nmf_sep.get_rest_mask(mask_dict)

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

    # r_spectra = np.zeros_like(spectra)
    # for key, audio in audio_dict.items():
    #     r_spectra += stft.calculatr_stft(
    #         audio,
    #         frame_size,
    #         hop_size
    #     )
    # r_spectra = sum(spectra_dict.values())
    # reconstructed_audio = stft.calculate_istft(
    #     r_spectra,
    #     frame_size,
    #     hop_size
    # )

    reconstructed_audio = sum(audio_dict.values())
    output_file = audio_filename("reconstructed")
    utils.save_audio(
        output_file,
        sample_rate,
        reconstructed_audio
    )

    non_vocal_audio = sum(audio_dict[key] for key in audio_dict if key != "vocal")
    output_file = audio_filename("non_vocal")
    utils.save_audio(
        output_file,
        sample_rate,
        non_vocal_audio
    )

    utils.save_nmf_analysis(
        df,
        output_file="Spectrograms/nmf/nmf_analysis.csv"
    )

    # Make audio from all components
    utils.save_all_component_audio(
        B, G, spectra,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        output_dir="Audio/nmf"
    )
    print()

    print("Done!")
    print("Original:", input_file)
    print("Reconstructed:", output_file)
    print("Spectrogram:", spectogram_filename())

if __name__ == "__main__":
    main()